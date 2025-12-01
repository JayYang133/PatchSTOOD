import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import numpy as np
import configparser
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from models.model import SqLinear
from lib.utils import log_string, loadData, _compute_loss, metric

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        log_string(log, '\n------------ Loading Data -------------')
        self.trainX, self.trainY, self.trainXTE, self.trainYTE,\
        self.valX, self.valY, self.valXTE, self.valYTE,\
        self.testX, self.testY, self.testXTE, self.testYTE,\
        self.mean, self.std,\
        self.partition_data = loadData(
                                self.traffic_file, self.meta_file,
                                self.input_len, self.output_len,
                                self.train_ratio, self.test_ratio,
                                self.adj_file,
                                self.tod, self.dow,
                                self.spa_patchsize, log, self.max_increase_ratio, self.istest, self.test_increase_ratio, self.test_decrease_ratio)
        # 训练集需要扰动，需要在train()中分配
        self.train_partitioner = self.partition_data['train_partitioner']
        self.train_nodes = self.partition_data['train_nodes']
        self.capacity = self.partition_data['capacity']
        # 验证集与测试集不需要空间扰动，可以在初始化时就分配好
        log_string(log, 'Generating static VAL patches...')
        self.val_ori_parts_idx, self.val_reo_parts_idx, self.val_reo_all_idx = \
            self.partition_data['val_partitioner'].get_patch_data(
                self.capacity, self.partition_data['val_nodes'])
        
        log_string(log, 'Generating static TEST patches...')
        self.test_ori_parts_idx, self.test_reo_parts_idx, self.test_reo_all_idx = \
            self.partition_data['test_partitioner'].get_patch_data(
                self.capacity, self.partition_data['test_nodes'])

        log_string(log, '------------ End -------------\n')

        self.best_epoch = 0

        self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        self.build_model()
    
    def build_model(self):                   
        self.model = SqLinear(self.tem_patchsize, self.tem_patchnum,
                            self.spa_patchsize, self.spa_patchnum,
                            self.tod, self.dow,
                            self.layers,
                            self.input_dims, self.node_dims, self.tod_dims, self.dow_dims,
                            mlp_ratio=self.mlp_ratio).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=self.learning_rate,weight_decay=self.weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.patience,
            verbose=True
        )

        log_string(log, '------------ Parameters -------------\n')
        log_string(log, f'Parameters: {self.model.get_n_param():,}')
        log_string(log, '------------ End -------------\n\n')

    def vali(self):
        self.model.eval()
        num_val = self.valX.shape[0]
        pred = []
        label = []

        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                if isinstance(self.model, torch.nn.Module):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * self.batch_size)

                    X = self.valX[start_idx : end_idx]
                    Y = self.valY[start_idx : end_idx]
                    TE = torch.from_numpy(self.valXTE[start_idx : end_idx]).to(self.device)
                    NormX = torch.from_numpy((X-self.mean)/self.std).float().to(self.device)

                    y_hat = self.model(NormX,TE,self.val_ori_parts_idx, self.val_reo_parts_idx, self.val_reo_all_idx) ###

                    pred.append(y_hat.cpu().numpy()*self.std+self.mean)
                    label.append(Y)
        
        pred = np.concatenate(pred, axis = 0)
        label = np.concatenate(label, axis = 0)

        maes = []
        rmses = []
        mapes = []

        for i in range(pred.shape[1]):
            mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)

    def train(self):
        log_string(log, "======================TRAIN MODE======================")
        min_loss = 10000000.0
        num_train = self.trainX.shape[0]

        current_lr = self.learning_rate
        decay_epochs = [] 

        for epoch in tqdm(range(1,self.max_epoch+1)):
            self.model.train()
            train_l_sum, train_acc_sum, batch_count, start = 0.0, 0.0, 0, time.time()
            # 每个epoch进行一次随机空间扰动
            patch_train_epoch = self.train_partitioner.get_perturbed_patch_data(
                        capacity=self.capacity,
                        current_nodes_ordered=self.train_nodes,
                        perturb_strategy=self.perturb_strategy,  
                        leaf_drop_ratio=self.leaf_drop_ratio
                    )
            train_ori_parts_idx_epoch, train_reo_parts_idx_epoch, train_reo_all_idx_epoch = patch_train_epoch # 分配扰动后的训练数据

            permutation = np.random.permutation(num_train)
            self.trainX = self.trainX[permutation]
            self.trainY = self.trainY[permutation]
            self.trainXTE = self.trainXTE[permutation]
            num_batch = math.ceil(num_train / self.batch_size)
            with tqdm(total=num_batch) as pbar:
                for batch_idx in range(num_batch):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_train, (batch_idx + 1) * self.batch_size)

                    X = self.trainX[start_idx : end_idx]
                    Y = self.trainY[start_idx : end_idx]
                    TE = torch.from_numpy(self.trainXTE[start_idx : end_idx]).to(self.device)
                    NormX = torch.from_numpy((X-self.mean)/self.std).float().to(self.device)
                    
                    Y = torch.from_numpy(Y).float().to(self.device)
                    
                    self.optimizer.zero_grad()

                    y_hat = self.model(NormX, TE, 
                                       train_ori_parts_idx_epoch, 
                                       train_reo_parts_idx_epoch, 
                                       train_reo_all_idx_epoch)

                    loss = _compute_loss(Y, y_hat*self.std+self.mean)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    
                    train_l_sum += loss.cpu().item()

                    batch_count += 1
                    pbar.update(1)
            log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
                % (epoch, self.optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))
            mae, rmse, mape = self.vali()
            self.lr_scheduler.step(mae[-1])

            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                log_string(log, f'Learning rate reduced at epoch {epoch}. New LR: {new_lr:.6f}')
                decay_epochs.append(epoch) 
                current_lr = new_lr

            if mae[-1] < min_loss:
                self.best_epoch = epoch
                min_loss = mae[-1]
                torch.save(self.model.state_dict(), self.model_file)
        
        log_string(log, f'Best epoch is: {self.best_epoch}')

    def test(self):
        log_string(log, "======================TEST MODE======================")
        self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        self.model.eval()
        num_val = self.testX.shape[0]
        pred = []
        label = []
        total_inference_time = 0.0
        total_samples = 0

        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                if isinstance(self.model, torch.nn.Module):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * self.batch_size)

                    X = self.testX[start_idx : end_idx]
                    Y = self.testY[start_idx : end_idx]
                    TE = torch.from_numpy(self.testXTE[start_idx : end_idx]).to(self.device)
                    NormX = torch.from_numpy((X-self.mean)/self.std).float().to(self.device)

                    
                    inference_start = time.time()
                    y_hat = self.model(NormX,TE,self.test_ori_parts_idx, self.test_reo_parts_idx, self.test_reo_all_idx)
                
                    inference_end = time.time()
                    
                    batch_inference_time = inference_end - inference_start
                    total_inference_time += batch_inference_time
                    total_samples += X.shape[0]

                    pred.append(y_hat.cpu().numpy()*self.std+self.mean)
                    label.append(Y)
        
        pred = np.concatenate(pred, axis = 0)
        label = np.concatenate(label, axis = 0)

        maes = []
        rmses = []
        mapes = []

        for i in range(pred.shape[1]):
            mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='configuration file')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
    parser.add_argument('--seed', type = int, default = config['train']['seed'])
    parser.add_argument('--batch_size', type = int, default = config['train']['batch_size'])
    parser.add_argument('--max_epoch', type = int, default = config['train']['max_epoch'])
    parser.add_argument('--learning_rate', type=float, default = config['train']['learning_rate'])
    parser.add_argument('--weight_decay', type=float, default = config['train']['weight_decay'])
    parser.add_argument('--patience', type=int, default = config['train']['patience'])
    parser.add_argument('--istest', type = str, default = config['train']['istest'])
    parser.add_argument('--max_increase_ratio', type = float, default = config['train']['max_increase_ratio'])
    parser.add_argument('--test_increase_ratio', type = float, default = config['train']['test_increase_ratio'])
    parser.add_argument('--test_decrease_ratio', type = float, default = config['train']['test_decrease_ratio'])  
    parser.add_argument('--perturb_strategy', type=str, default=config['train']['perturb_strategy'])
    parser.add_argument('--leaf_drop_ratio', type=float, default=config['train']['leaf_drop_ratio'])

    parser.add_argument('--input_len', type = int, default = config['data']['input_len'])
    parser.add_argument('--output_len', type = int, default = config['data']['output_len'])
    parser.add_argument('--train_ratio', type = float, default = config['data']['train_ratio'])
    parser.add_argument('--val_ratio', type = float, default = config['data']['val_ratio'])
    parser.add_argument('--test_ratio', type = float, default = config['data']['test_ratio'])

    parser.add_argument('--layers', type=int, default = config['param']['layers'])
    parser.add_argument('--tem_patchsize', type = int, default = config['param']['tps'])
    parser.add_argument('--tem_patchnum', type = int, default = config['param']['tpn'])
    parser.add_argument('--spa_patchsize', type = int, default = config['param']['capacity'])
    parser.add_argument('--spa_patchnum', type = int, default = config['param']['patchnum'])
    parser.add_argument('--node_num', type = int, default = config['param']['nodes'])
    parser.add_argument('--tod', type=int, default = config['param']['tod'])
    parser.add_argument('--dow', type=int, default = config['param']['dow'])
    parser.add_argument('--input_dims', type=int, default = config['param']['id'])
    parser.add_argument('--node_dims', type=int, default = config['param']['nd'])
    parser.add_argument('--tod_dims', type=int, default = config['param']['td'])
    parser.add_argument('--dow_dims', type=int, default = config['param']['dd'])
    parser.add_argument('--mlp_ratio', type=float, default = config['param']['mlp_ratio'])
    parser.add_argument('--rank', type=int, default = config['param']['rank'])

    parser.add_argument('--traffic_file', default = config['file']['traffic'])
    parser.add_argument('--meta_file', default = config['file']['meta'])
    parser.add_argument('--adj_file', default = config['file']['adj'])
    parser.add_argument('--model_file', default = config['file']['model'])
    parser.add_argument('--log_file', default = config['file']['log'])

    args = parser.parse_args()

    log = open(args.log_file, 'w')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    log_string(log, '------------ Options -------------')
    for k, v in vars(args).items():
        log_string(log, '%s: %s' % (str(k), str(v)))
    log_string(log, '-------------- End ----------------')

    solver = Solver(vars(args))
    
    if args.istest == 'False':
        solver.train()
    else:
        solver.test()