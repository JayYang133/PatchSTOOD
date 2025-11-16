import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from lib.DynamicSpatialPartitioner import DynamicSpatialPartitioner

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        wape = np.divide(np.sum(mae), np.sum(label))
        wape = np.nan_to_num(wape * mask)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def _compute_loss(y_true, y_predicted):
        return masked_mae(y_predicted, y_true, 0.0)

def seq2instance(data, P, Q):
    num_step, nodes, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, nodes, dims))
    #y = np.zeros(shape = (num_sample, P, nodes, dims)) ##########
    y = np.zeros(shape=(num_sample, Q, nodes, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + Q : i + P + Q]
    return x, y

def read_meta(path):
    meta = pd.read_csv(path)
    lat = meta['Lat'].values
    lng = meta['Lng'].values
    locations = np.stack([lat,lng], 0)
    return locations

def construct_adj(data, num_node):
    # construct the adj through the cosine similarity
    data_mean = np.mean([data[24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
    data_mean = data_mean.squeeze().T
    tem_matrix = cosine_similarity(data_mean, data_mean)
    tem_matrix = np.exp((tem_matrix-tem_matrix.mean())/tem_matrix.std())
    return tem_matrix

def augmentAlign(dist_matrix, auglen):
    # find the most similar points in other leaf nodes
    sorted_idx = np.argsort(dist_matrix.reshape(-1)*-1)
    sorted_idx = sorted_idx % dist_matrix.shape[-1]
    augidx = []
    for idx in sorted_idx:
        if idx not in augidx:
            augidx.append(idx)
        if len(augidx) == auglen:
            break
    return np.array(augidx, dtype=int)

def loadData(filepath, metapath, P, Q, train_ratio, test_ratio, adjpath, tod, dow, capacity, log, 
             new_node_ratio=0.1, istest=False, test_ratio_ood=0.1):

    Traffic = np.load(filepath)['data'][...,:1]
    locations = read_meta(metapath)
    num_step = Traffic.shape[0]
    # temporal positions
    TE = np.zeros([num_step, 2])
    TE[:,0] = np.array([i % tod for i in range(num_step)])
    TE[:,1] = np.array([(i // tod) % dow for i in range(num_step)])
    TE_tile = np.repeat(np.expand_dims(TE, 1), Traffic.shape[1], 1)
    log_string(log, f'Shape of data: {Traffic.shape}')
    log_string(log, f'Shape of locations: {locations.shape}')
    # train/val/test 
    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    
    trainData, trainTE = Traffic[: train_steps], TE_tile[: train_steps]
    valData, valTE = Traffic[train_steps : train_steps + val_steps], TE_tile[train_steps : train_steps + val_steps]
    testData, testTE = Traffic[-test_steps :], TE_tile[-test_steps :]

    if os.path.exists(adjpath):
        adj = np.load(adjpath)
    else:
        adj = construct_adj(trainData, locations.shape[1])
        np.save(adjpath, adj)

    # ==================== OOD 节点处理 ====================
    N = Traffic.shape[1]
    nodes = np.arange(N)
    np.random.shuffle(nodes)
    num_fixed = min(int(N / (1 + 3 * new_node_ratio)), N - 3)# num_val = 1,num_fixed_node = min(int(self.num_nodes / (1 + (self.num_val + 2) * self.new_node_ratio)), \self.num_nodes - self.num_val - 2)
    num_per  = max(int(num_fixed * new_node_ratio), 1)
    fixed_idx = np.sort(nodes[:num_fixed])
    remain = nodes[num_fixed:]

    train_unobs = np.sort(remain[:num_per])
    val_unobs   = np.sort(remain[num_per:2*num_per])
    test_unobs  = np.sort(remain[2*num_per:3*num_per])
    extra = np.sort(remain[3*num_per:4*num_per])

    if istest:
        if test_ratio_ood == 0.05:
            test_unobs = test_unobs[:len(test_unobs)//2]
        elif test_ratio_ood == 0.15:
            test_unobs = np.concatenate([test_unobs, extra[:len(test_unobs)//2]])
        elif test_ratio_ood == 0.2:
            test_unobs = np.concatenate([test_unobs, extra])

    train_nodes = np.concatenate([fixed_idx, train_unobs])
    val_nodes   = np.concatenate([fixed_idx, val_unobs])
    test_nodes  = np.concatenate([fixed_idx, test_unobs])
    log_string(log, f'OOD: fixed={num_fixed}, per={num_per}, '
                   f'test_unobs={len(test_unobs)} (×{test_ratio_ood})')

    # ==================== patch划分与重构  ====================
    partitioner = DynamicSpatialPartitioner(all_points=locations.T, 
                                            capacity=capacity)

    log_string(log, 'Building initial partition for TRAIN...')
    partitioner.build(train_nodes)
    patch_train = partitioner.get_patch_data(capacity, train_nodes)
    log_string(log, f'Train patches: {len(patch_train[0])} nodes in {len(partitioner.get_patches())} patches')

    log_string(log, 'Updating partition for VAL...')
    val_partitioner = partitioner.deepcopy()
    val_partitioner.update_partition(set(val_nodes), log) 
    patch_val = val_partitioner.get_patch_data(capacity, val_nodes)
    log_string(log, f'Val patches: {len(patch_val[0])} nodes in {len(partitioner.get_patches())} patches')

    log_string(log, 'Updating partition for TEST...')
    test_partitioner = partitioner.deepcopy()
    test_partitioner.update_partition(set(test_nodes), log)
    patch_test = test_partitioner.get_patch_data(capacity, test_nodes)
    log_string(log, f'Test patches: {len(patch_test[0])} nodes in {len(partitioner.get_patches())} patches')

    trainTE_sub = trainTE[:, train_nodes, :]
    valTE_sub   = valTE[:,   val_nodes,   :]
    testTE_sub  = testTE[:,  test_nodes,  :]

    trainX, trainY = seq2instance(trainData[:, train_nodes], P, Q)
    valX,   valY   = seq2instance(valData[:,   val_nodes],   P, Q)
    testX,  testY  = seq2instance(testData[:,  test_nodes],  P, Q)
    trainXTE, trainYTE = seq2instance(trainTE_sub, P, Q)
    valXTE,   valYTE   = seq2instance(valTE_sub,   P, Q)
    testXTE,  testYTE  = seq2instance(testTE_sub, P, Q)

    mean, std = np.mean(trainX), np.std(trainX)

    log_string(log, f'Shape of Train: {trainY.shape}')
    log_string(log, f'Shape of Validation: {valY.shape}')
    log_string(log, f'Shape of Test: {testY.shape}')
    log_string(log, f'Mean: {mean} & Std: {std}')

    return trainX, trainY, trainXTE, trainYTE, valX, valY, valXTE, valYTE, testX, testY, testXTE, testYTE, mean, std, patch_train, patch_val, patch_test
