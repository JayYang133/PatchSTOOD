import os
import torch
import numpy as np
import pandas as pd

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
        self.capacity = capacity
        self.root = None

        self.node_to_leaf_map = {}

    def _recursive_build(self, indices, parent=None):
        """
        square_partition
        """
        if not indices:
            return None
            
        node = PartitionNode(indices, parent)
        
        sub_points = self.all_points[indices]
        min_xy = sub_points.min(axis=0)
        max_xy = sub_points.max(axis=0)
        node.bbox = (min_xy, max_xy)

        if len(indices) <= self.capacity:
            node.is_leaf = True
            for idx in indices:
                self.node_to_leaf_map[idx] = node
            return node

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
        
        left_child = self._recursive_build(list(left_indices), parent=node)
        right_child = self._recursive_build(list(right_indices), parent=node)
        node.children = [left_child, right_child]

        node.R_n = self._calculate_R_n(node)
        
        return node

    def build(self, initial_indices):
        self.node_to_leaf_map.clear()
        self.root = self._recursive_build(list(initial_indices))

    def _rebuild_region(self, node, log):
        # 重构        
        all_indices_in_region = self._get_all_indices_under(node)
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

        if node is None:
            node = self.root
            
        if node is None:
            return None

        if node.is_leaf:
            return node
        
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

        point = self.all_points[node_index]
        leaf = self._find_leaf_for_point(point)

        if leaf is None:
            self.build([node_index])
            return

        leaf.indices.append(node_index)
        self.node_to_leaf_map[node_index] = leaf

        if len(leaf.indices) > self.capacity:
            self._split_leaf(leaf, log)

    def _split_leaf(self, node, log):
        
        node.is_leaf = False
        current_indices = node.indices
        node.indices = [] 
        
        sub_points = self.all_points[current_indices]
        min_xy, max_xy = sub_points.min(axis=0), sub_points.max(axis=0)
        span = max_xy - min_xy
        node.split_axis = np.argmax(span)
        
        sorted_idx_local = np.argsort(sub_points[:, node.split_axis])
        sorted_indices_global = np.array(current_indices)[sorted_idx_local]
        
        split_idx_local = len(sorted_indices_global) // 2

        if split_idx_local == 0:
            split_idx_local = 1
        
        node.split_val = sub_points[sorted_idx_local[split_idx_local], node.split_axis]
        
        left_indices = sorted_indices_global[:split_idx_local]
        right_indices = sorted_indices_global[split_idx_local:]

        left_child = self._recursive_build(list(left_indices), parent=node)
        right_child = self._recursive_build(list(right_indices), parent=node)
        node.children = [left_child, right_child]

        node.R_n = self._calculate_R_n(node)

        self._propagate_split_metric(node.parent, log)

    def remove_node(self, node_index, log):

        if node_index not in self.node_to_leaf_map:
            return
            
        leaf = self.node_to_leaf_map.pop(node_index)

        if node_index in leaf.indices:
            leaf.indices.remove(node_index)

        if not leaf.indices:
            self._prune_leaf(leaf, log)

    def _prune_leaf(self, leaf, log):
        """
        删除空 Patch，并可能触发父节点合并
        """
        parent = leaf.parent
        if parent is None:
            if self.root == leaf:
                self.root = None
            return

        sibling = parent.children[0] if parent.children[1] == leaf else parent.children[1]
        grandparent = parent.parent
        if grandparent is None:
            self.root = sibling
            if sibling:
                sibling.parent = None
        else:
            if grandparent.children[0] == parent:
                grandparent.children[0] = sibling
            else:
                grandparent.children[1] = sibling
            if sibling:
                sibling.parent = grandparent
        
        if grandparent:
            self._check_rebuild(grandparent, log)

    def _propagate_split_metric(self, node, log):

        curr = node
        while curr is not None:
            curr.W_n += 1
            
            self._check_rebuild(curr, log)
            
            curr = curr.parent

    def _calculate_R_n(self, node):
        """
        最大跨度(对角线长)
        """
        if node.is_leaf or node.bbox is None:
            return 1.0
        
        span = node.bbox[1] - node.bbox[0]
        diag_span = np.linalg.norm(span)# 对角线长度
        return 1.0 / (diag_span + 1e-6)
        
    def _calculate_health(self, node):
        leaves = self._get_all_leaves_under(node)
        if not leaves:
            return 1.0
        
        total_nodes = sum(len(l.indices) for l in leaves)
        total_capacity = len(leaves) * self.capacity
        
        health = total_nodes / (total_capacity + 1e-6)
        return health

    def _check_rebuild(self, node, log):
        if node is None or node.is_leaf:
            return

        health = self._calculate_health(node)
        
        tolerance = node.W_n / (node.R_n + 1e-6)
        
        if health < tolerance:
            self._rebuild_region(node, log)

    def _get_all_leaves_under(self, node):
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
        indices = []
        leaves = self._get_all_leaves_under(node)
        for leaf in leaves:
            indices.extend(leaf.indices)
        return list(set(indices))

    def get_patches(self):
        leaves = self._get_all_leaves_under(self.root)
        return [l.indices for l in leaves if l.indices]

    def update_partition(self, new_node_set, log):
        current_node_set = set(self.node_to_leaf_map.keys())
        
        nodes_to_remove = current_node_set - new_node_set
        nodes_to_add = new_node_set - current_node_set
        
        for idx in nodes_to_remove:
            self.remove_node(idx, log)
            
        for idx in nodes_to_add:
            self.add_node(idx, log)

    def get_patch_data(self, capacity, current_nodes_ordered):
        parts_idx = self.get_patches()
        
        if not parts_idx:
            return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))

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