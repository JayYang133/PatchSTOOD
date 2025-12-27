import os
import torch
import numpy as np
import pandas as pd
import random
import secrets

def reorderData(parts_idx, mxlen, adj, capacity):
    ori_parts_idx = np.array([], dtype=int)
    reo_parts_idx = np.array([], dtype=int)
    reo_all_idx = np.array([], dtype=int)
    
    current_offset = 0
    for i, part_idx in enumerate(parts_idx):
        part_size = len(part_idx)
        local_reo_idx = np.arange(part_size) + current_offset
        
        reo_parts_idx = np.concatenate([reo_parts_idx, local_reo_idx])
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
        reo_all_idx = np.concatenate([reo_all_idx, part_idx])
        
        current_offset += part_size

    return ori_parts_idx, reo_parts_idx, reo_all_idx

class PartitionNode:
    """
    分区树节点(内部节点或叶子节点)
    """
    def __init__(self, indices, parent=None, is_leaf=False):
        self.indices = list(indices) 
        self.parent = parent         
        self.children = []           
        self.is_leaf = is_leaf      
        
        self.bbox = None             # [min_xy, max_xy]
        self.split_axis = None       # 分裂轴 (0, 1)
        self.split_val = None        # 分裂值
        
        self.W_n = 0                 
        self.R_n = 0.0               

class DynamicSpatialPartitioner:
    def __init__(self, all_points, capacity=8):

        self.all_points = all_points # [lng, lat]
        self.capacity = capacity # 每个分区的最大节点数
        self.root = None # 根节点

        self.node_to_leaf_map = {} # 映射：节点索引 -> 所在的叶子节点

    def _recursive_build(self, indices, parent=None):
        """
        递归构建分区树
        原square_partition
        """
        if not indices:
            return None
            
        node = PartitionNode(indices, parent)
        
        sub_points = self.all_points[indices]
        min_xy = sub_points.min(axis=0)
        max_xy = sub_points.max(axis=0)
        node.bbox = (min_xy, max_xy)
        # 终止:节点数量小于等于容量
        if len(indices) <= self.capacity:
            node.is_leaf = True
            for idx in indices:
                self.node_to_leaf_map[idx] = node
            return node
        # 沿着跨度最大的维度分裂
        span = max_xy - min_xy
        node.split_axis = np.argmax(span)
        
        sorted_idx_local = np.argsort(sub_points[:, node.split_axis])
        sorted_indices_global = np.array(indices)[sorted_idx_local]
        
        N = len(sorted_indices_global)
        candidates = [i for i in range(self.capacity, N, self.capacity)]
        if not candidates:
            split_idx_local = N // 2
        else:
            mid = N // 2
            split_idx_local = min(candidates, key=lambda x: abs(x - mid))
        
        node.split_val = sub_points[sorted_idx_local[split_idx_local], node.split_axis]
        
        left_indices = sorted_indices_global[:split_idx_local]
        right_indices = sorted_indices_global[split_idx_local:]
        # 归构建左右子树
        left_child = self._recursive_build(list(left_indices), parent=node)
        right_child = self._recursive_build(list(right_indices), parent=node)
        node.children = [left_child, right_child]
        # 每个节点计算R_n
        node.R_n = self._calculate_R_n(node)
        
        return node

    def build(self, initial_indices):
        self.node_to_leaf_map.clear()
        self.root = self._recursive_build(list(initial_indices))

    def _rebuild_region(self, node, log):
        """
        重构区域
        """   
        all_indices_in_region = self._get_all_indices_under(node) # 获取该节点下的所有索引
        if not all_indices_in_region:
            return

        parent = node.parent
        
        new_subtree_root = self._recursive_build(all_indices_in_region, parent=parent)
        
        if parent is None:
            self.root = new_subtree_root 
        elif parent.children[0] == node:
            parent.children[0] = new_subtree_root
        else:
            parent.children[1] = new_subtree_root
            
        if new_subtree_root:
            new_subtree_root.W_n = 0

    def _find_leaf_for_point(self, point, node=None):
        # 为给定点寻找对应的叶子节点(patch)
        if node is None:
            node = self.root
            
        if node is None:
            return None

        if node.is_leaf:
            return node
        # 根据分裂条件选择搜索路径
        if point[node.split_axis] < node.split_val:
            if node.children[0]:
                return self._find_leaf_for_point(point, node.children[0])
            else:
                return self._find_leaf_for_point(point, node.children[1] if node.children[1] else node)
        else:
            if node.children[1]:
                return self._find_leaf_for_point(point, node.children[1])
            else:
                return self._find_leaf_for_point(point, node.children[0] if node.children[0] else node)


    def add_node(self, node_index, log):
        """
        添加新节点到分区树中
        """
        point = self.all_points[node_index] # 获取节点坐标
        leaf = self._find_leaf_for_point(point) # 找到对应的叶子节点(对应的patch)

        if leaf is None:
            self.build([node_index])
            return
        # 将节点添加到叶子中
        leaf.indices.append(node_index)
        self.node_to_leaf_map[node_index] = leaf
        # 超过容量需要分裂
        if len(leaf.indices) > self.capacity:
            self._split_leaf(leaf, log)

    def _split_leaf(self, node, log):
        """
        添加新节点到分区树中，超过容量需要分裂
        """
        node.is_leaf = False # 将叶子节点转为内部节点
        current_indices = node.indices
        node.indices = [] 
        
        sub_points = self.all_points[current_indices]
        min_xy, max_xy = sub_points.min(axis=0), sub_points.max(axis=0)
        span = max_xy - min_xy
        node.split_axis = np.argmax(span)

        # 排序并选择分割点
        sorted_idx_local = np.argsort(sub_points[:, node.split_axis])
        sorted_indices_global = np.array(current_indices)[sorted_idx_local]
        split_idx_local = len(sorted_indices_global) // 2 # 中点分割

        if split_idx_local == 0:
            split_idx_local = 1
        
        node.split_val = sub_points[sorted_idx_local[split_idx_local], node.split_axis]
        # 创建左右子节点
        left_indices = sorted_indices_global[:split_idx_local]
        right_indices = sorted_indices_global[split_idx_local:]

        left_child = self._recursive_build(list(left_indices), parent=node)
        right_child = self._recursive_build(list(right_indices), parent=node)
        node.children = [left_child, right_child]
        # 更新区域度量
        node.R_n = self._calculate_R_n(node)
        # 向上传播分裂信息，可能触发重构
        self._propagate_split_metric(node.parent, log)

    def remove_node(self, node_index, log):
        """
        从分区树中移除节点
        """
        if node_index not in self.node_to_leaf_map:
            return
            
        leaf = self.node_to_leaf_map.pop(node_index) # 从映射中移除

        if node_index in leaf.indices:
            leaf.indices.remove(node_index) # 从叶子节点的索引列表中移除

        if not leaf.indices:
            self._prune_leaf(leaf, log) # 如果叶子(patch)变空，进行剪枝

    def _prune_leaf(self, leaf, log):
        """
        剪枝空叶子节点 - 删除空叶子并可能合并父节点
        """
        parent = leaf.parent
        if parent is None:
            if self.root == leaf:
                self.root = None
            return
        # 找到兄弟节点（父节点的另一个子节点）
        sibling = parent.children[0] if parent.children[1] == leaf else parent.children[1]
        grandparent = parent.parent
        if grandparent is None:
            # 父节点是根节点的情况
            self.root = sibling
            if sibling:
                sibling.parent = None
        else:
            # 否则用兄弟节点替换父节点
            if grandparent.children[0] == parent:
                grandparent.children[0] = sibling
            else:
                grandparent.children[1] = sibling
            if sibling:
                sibling.parent = grandparent
        # 检查是否需要重构
        if grandparent:
            self._check_rebuild(grandparent, log)

    def _propagate_split_metric(self, node, log):
        """
        向上传播分裂度量 - 更新祖先节点的分裂权重       
        当节点发生分裂时，需要向上更新所有祖先节点的W_n值
        """
        curr = node
        while curr is not None:
            curr.W_n += 1
            
            self._check_rebuild(curr, log)
            
            curr = curr.parent

    def _calculate_R_n(self, node):
        """
        区域范围度量：最大跨度(对角线长)
        """
        if node.is_leaf or node.bbox is None:
            return 1.0
        
        span = node.bbox[1] - node.bbox[0]
        diag_span = np.linalg.norm(span)# 对角线长度
        return 1.0 / (diag_span + 1e-6)
        
    def _calculate_health(self, node):
        """
        分区健康度
        实际节点数 / 理论最大容量(填充率)
        """
        leaves = self._get_all_leaves_under(node)
        if not leaves:
            return 1.0
        
        total_nodes = sum(len(l.indices) for l in leaves)
        total_capacity = len(leaves) * self.capacity
        
        health = total_nodes / (total_capacity + 1e-6)
        return health

    def _check_rebuild(self, node, log):
        """
        检查是否需要重构节点
        健康度较低且分裂频繁的区域需要重构
        """
        if node is None or node.is_leaf:
            return

        health = self._calculate_health(node)
        
        tolerance = node.W_n / (node.R_n + 1e-6)
        
        if health < tolerance:
            self._rebuild_region(node, log)

    def _get_all_leaves_under(self, node):
        """
        获取节点下的所有叶子节点
        """
        leaves = []
        if node is None:
            return leaves
        
        q = [node]
        while q:
            curr = q.pop(0)
            if curr.is_leaf:
                leaves.append(curr)
            else:
                for child in curr.children:
                    if child:
                        q.append(child)
        return leaves

    def _get_all_indices_under(self, node):
        """
        获取节点下的所有索引
        """
        indices = []
        leaves = self._get_all_leaves_under(node)
        for leaf in leaves:
            indices.extend(leaf.indices)
        return list(set(indices))

    def get_patches(self):
        """
        获取当前所有分区（非空叶子节点）
        """
        leaves = self._get_all_leaves_under(self.root) # 所有叶子
        return [l.indices for l in leaves if l.indices] # 返回非空分区的索引列表

    def update_partition(self, new_node_set, log):
        """
        更新分区结构        
        1. 移除不再存在的节点
        2. 添加新出现的节点
        """
        current_node_set = set(self.node_to_leaf_map.keys())
        
        nodes_to_remove = current_node_set - new_node_set
        nodes_to_add = new_node_set - current_node_set
        
        for idx in nodes_to_remove:
            self.remove_node(idx, log)
            
        for idx in nodes_to_add:
            self.add_node(idx, log)

    # 无padding
    '''def get_patch_data_without_padding(self, capacity, current_nodes_ordered):
        """
        获取分区数据
    
        将分区索引转换为模型需要的格式，包括：
        - 全局索引到局部索引的映射
        - 重新排序的索引数组
        """
        parts_idx = self.get_patches() # 获取所有分区
        
        if not parts_idx:
            return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))
        # 创建全局索引到局部索引的映射
        global_to_local_map = {
            global_idx: local_idx 
            for local_idx, global_idx in enumerate(current_nodes_ordered)
        }
        # 将全局索引转换为局部索引
        local_parts_idx = []
        for p_list in parts_idx:
            local_patch_filtered = []
            for g_idx in p_list:
                local_idx = global_to_local_map.get(g_idx) 
                if local_idx is not None:
                    local_patch_filtered.append(local_idx)
            
            if local_patch_filtered:
                local_parts_idx.append(local_patch_filtered)
        
        if not local_parts_idx:
            return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))
        # 重排数据(原SqLinear)
        mxlen = max(len(p) for p in local_parts_idx)
        ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(local_parts_idx, mxlen, None, capacity) 
        
        return (ori_parts_idx, reo_parts_idx, reo_all_idx)'''
    
    # 这个版本是第一版含padding的，理论上会造成最后一个patch 的padding节点过多
    '''def get_patch_data_version1(self, capacity, current_nodes_ordered):
        """
        获取 Patch 数据，返回相对于当前节点子集的局部重排序索引。
        """
        parts_idx = self.get_patches() # 返回全局索引
        
        if not parts_idx:
            return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))

        global_to_local_map = {
            global_idx: local_idx 
            for local_idx, global_idx in enumerate(current_nodes_ordered)
        }

        # 2. 将 parts_idx (全局索引) 转换为局部索引
        local_parts_idx = []
        for p_list in parts_idx:
            local_patch_filtered = []
            for g_idx in p_list:
                local_idx = global_to_local_map.get(g_idx) 
                if local_idx is not None: 
                    local_patch_filtered.append(local_idx)
            
            if local_patch_filtered:
                local_parts_idx.append(local_patch_filtered)
        
        if not local_parts_idx:
            return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))

        # 3. 合并所有节点并重新分配到固定大小的patches
        all_nodes = []
        for patch in local_parts_idx:
            all_nodes.extend(patch)
        
        total_nodes = len(all_nodes)
        num_patches = (total_nodes + capacity - 1) // capacity
        
        balanced_patches = []
        current_idx = 0
        
        for i in range(num_patches):
            # 为每个patch分配capacity个节点
            patch_size = capacity
            
            # 获取实际节点
            patch_nodes = all_nodes[current_idx:current_idx+patch_size]
            
            # 如果节点数不足，使用padding（重复最后一个节点）
            if len(patch_nodes) < patch_size:
                if patch_nodes:  # 如果有实际节点，使用最后一个节点进行padding
                    padding = [patch_nodes[-1]] * (patch_size - len(patch_nodes))
                    patch_nodes.extend(padding)

            
            balanced_patches.append(patch_nodes)
            current_idx += patch_size

        mxlen = capacity 
        ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(balanced_patches, mxlen, None, capacity) 
        
        return (ori_parts_idx, reo_parts_idx, reo_all_idx)'''

    # 这个版本是第二版，对每个patch独立padding
    def get_patch_data(self, capacity, current_nodes_ordered):
        """
        获取 Patch 数据，返回相对于当前节点子集的局部重排序索引。
        """
        parts_idx = self.get_patches() # 返回全局索引
        
        if not parts_idx:
            return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))

        global_to_local_map = {
            global_idx: local_idx 
            for local_idx, global_idx in enumerate(current_nodes_ordered)
        }

        # 2. 将 parts_idx (全局索引) 转换为局部索引
        local_parts_idx = []
        for p_list in parts_idx:
            local_patch_filtered = []
            for g_idx in p_list:
                local_idx = global_to_local_map.get(g_idx) 
                if local_idx is not None: 
                    local_patch_filtered.append(local_idx)
            
            if local_patch_filtered:
                local_parts_idx.append(local_patch_filtered)
        
        if not local_parts_idx:
            return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))

        # 3. 对每个patch独立处理，使用最后一个元素padding
        balanced_patches = []
        
        for patch in local_parts_idx:
            if not patch:  
                continue
            padded_patch = list(patch)  
            
            if len(padded_patch) < capacity:
                # 用最后一个节点填充
                last_node = padded_patch[-1] if padded_patch else 0
                while len(padded_patch) < capacity:
                    padded_patch.append(last_node)
            
            balanced_patches.append(padded_patch)

        mxlen = capacity 
        ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(balanced_patches, mxlen, None, capacity) 
        
        return (ori_parts_idx, reo_parts_idx, reo_all_idx)

    def _deep_copy_node(self, node, parent_map, node_map):
        if node is None:
            return None
            
        new_node = PartitionNode(
            indices=list(node.indices),
            parent=None,
            is_leaf=node.is_leaf
        )
        
        new_node.bbox = node.bbox
        new_node.split_axis = node.split_axis
        new_node.split_val = node.split_val

        new_node.W_n = node.W_n
        new_node.R_n = node.R_n

        parent_map[node] = new_node
        
        new_node.children = [
            self._deep_copy_node(child, parent_map, node_map)
            for child in node.children
        ]

        if new_node.is_leaf:
            for idx in new_node.indices:
                node_map[idx] = new_node
        
        return new_node

    def deepcopy(self):
        """
        避免val和test重构影响train的树，需要拷贝一棵树进行重构
        """
        new_partitioner = DynamicSpatialPartitioner(
            all_points=self.all_points, 
            capacity=self.capacity,
        )

        node_map = {}
        parent_map = {}

        new_partitioner.root = self._deep_copy_node(self.root, parent_map, node_map)

        q = [new_partitioner.root]
        while q:
            curr = q.pop(0)
            if curr:
                for i, child in enumerate(curr.children):
                    if child:
                        child.parent = curr
                        q.append(child)

        new_partitioner.node_to_leaf_map = node_map
        
        return new_partitioner

    # 无padding版本
    '''def get_perturbed_patch_data_without_padding(self, capacity, current_nodes_ordered,
                                 perturb_strategy='both', 
                                 leaf_drop_ratio=0.1,
                                 max_mask_ratio=0.2):
        """
        空间索引扰动
        leaf_drop_ratio为随机剪枝的剪枝率
        max_mask_ratio为子树遮蔽的最大遮蔽率
        """
        # 获取所有叶子节点(patch)
        all_leaves = self._get_all_leaves_under(self.root)
        total_leaves_count = len(all_leaves)
        
        perturbed_leaves = list(all_leaves) 
        if not perturbed_leaves:
             return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))

        # 选择扰动策略
        effective_strategy = perturb_strategy
        if perturb_strategy == 'both':
            effective_strategy = secrets.choice(['leaf_drop', 'subtree_mask'])
        # 随机剪枝
        if effective_strategy == 'leaf_drop' and len(perturbed_leaves) > 1:
            num_to_keep = int(len(perturbed_leaves) * (1.0 - leaf_drop_ratio))
            perturbed_leaves = random.sample(perturbed_leaves, k=num_to_keep)# 随机保留1.0 - leaf_drop_ratio的叶子结点
        # 子树遮蔽
        elif effective_strategy == 'subtree_mask':
            candidate_nodes = []
            # 搜索寻找合适的子树(含节点max_mask_ratio以下)
            q = [self.root]
            while q:
                curr = q.pop(0)
                if curr and not curr.is_leaf:
                    
                    leaves_under = self._get_all_leaves_under(curr)
                    ratio = len(leaves_under) / (total_leaves_count + 1e-6)

                    if 0.0 < ratio <= max_mask_ratio:
                        candidate_nodes.append((curr, leaves_under))

                    for child in curr.children:
                        q.append(child)
            # 如果有候选子树，随机选择一个进行屏蔽
            if candidate_nodes:
                node_to_mask, masked_leaves = random.choice(candidate_nodes)
                masked_set = set(masked_leaves)
                
                perturbed_leaves = [leaf for leaf in perturbed_leaves if leaf not in masked_set] # 从扰动叶子中移除被屏蔽的叶子
            else:
                # 回退到随机剪枝
                num_to_keep = int(len(perturbed_leaves) * 0.9)
                perturbed_leaves = random.sample(perturbed_leaves, k=num_to_keep)

        parts_idx = [l.indices for l in perturbed_leaves if l.indices]
        
        if not parts_idx:
             return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))
        # 全局到局部索引转换
        global_to_local_map = {
            global_idx: local_idx 
            for local_idx, global_idx in enumerate(current_nodes_ordered)
        }
 
        local_parts_idx = []
        for p_list in parts_idx: 
            local_patch_filtered = []
            for g_idx in p_list:
                local_idx = global_to_local_map.get(g_idx) 
                if local_idx is not None:
                    local_patch_filtered.append(local_idx)
           
            if local_patch_filtered:
                local_parts_idx.append(local_patch_filtered)
       
        if not local_parts_idx:
            return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))
 
        mxlen = max(len(p) for p in local_parts_idx)
        
        ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(local_parts_idx, mxlen, None, capacity) 
        
        return (ori_parts_idx, reo_parts_idx, reo_all_idx)'''

    # 含padding版本，对每个patch独立padding
    def get_perturbed_patch_data(self, capacity, current_nodes_ordered,
                                 perturb_strategy='both', 
                                 leaf_drop_ratio=0.1,
                                 max_mask_ratio=0.2):
        """
        获取扰动后的 Patch 数据，用于训练时的空间索引扰动。
        """

        # 1. 获取所有叶子节点
        all_leaves = self._get_all_leaves_under(self.root)
        total_leaves_count = len(all_leaves)
        
        perturbed_leaves = list(all_leaves) 
        if not perturbed_leaves:
             return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))

        effective_strategy = perturb_strategy
        if perturb_strategy == 'both':
            effective_strategy = secrets.choice(['subtree_mask', 'leaf_drop'])
        
        # --- 策略 1: 随机叶节点丢弃 ---
        if effective_strategy == 'leaf_drop' and len(perturbed_leaves) > 1:
            num_to_keep = int(len(perturbed_leaves) * (1.0 - leaf_drop_ratio))
            perturbed_leaves = random.sample(perturbed_leaves, k=num_to_keep)
        
        # --- 策略 2: 子树遮蔽 ---
        elif effective_strategy == 'subtree_mask':
            # 收集所有符合“规模约束”的内部节点
            candidate_nodes = []
            
            # 遍历所有节点
            q = [self.root]
            while q:
                curr = q.pop(0)
                if curr and not curr.is_leaf:
                    # 检查该节点下的叶子数量
                    leaves_under = self._get_all_leaves_under(curr)
                    ratio = len(leaves_under) / (total_leaves_count + 1e-6)
                    
                    # 只有当遮蔽比例小于阈值时，才作为候选
                    if 0.0 < ratio <= max_mask_ratio:
                        candidate_nodes.append((curr, leaves_under))
                    
                    # 继续遍历子节点
                    for child in curr.children:
                        q.append(child)
            
            if candidate_nodes:
                # 随机选择一个符合条件的子树
                node_to_mask, masked_leaves = random.choice(candidate_nodes)
                masked_set = set(masked_leaves)
                
                # 移除
                perturbed_leaves = [leaf for leaf in perturbed_leaves if leaf not in masked_set]
            else:
                # 如果没有符合条件的子树，退化为 leaf_drop
                num_to_keep = int(len(perturbed_leaves) * 0.9)
                perturbed_leaves = random.sample(perturbed_leaves, k=num_to_keep)

        # 4. 从扰动后的叶子节点中提取全局索引，形成分区索引列表
        parts_idx = [l.indices for l in perturbed_leaves if l.indices]

        if not parts_idx:
             return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))
 
        global_to_local_map = {
            global_idx: local_idx 
            for local_idx, global_idx in enumerate(current_nodes_ordered)
        }
        # 5. padding
        local_parts_idx = []
        for p_list in parts_idx: 
            local_patch_filtered = []
            for g_idx in p_list:
                local_idx = global_to_local_map.get(g_idx) 
                if local_idx is not None:
                    local_patch_filtered.append(local_idx)
           
            if local_patch_filtered:
                # 如果patch的大小小于capacity，使用当前patch的最后一个节点进行padding
                while len(local_patch_filtered) < capacity:
                    local_patch_filtered.append(local_patch_filtered[-1])
                local_parts_idx.append(local_patch_filtered)
       
        if not local_parts_idx:
            return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))
 
        mxlen = capacity
        ori_parts_idx, reo_parts_idx, reo_all_idx = reorderData(local_parts_idx, mxlen, None, capacity) 
        
        return (ori_parts_idx, reo_parts_idx, reo_all_idx)