import torch
import torch.nn as nn
import torch.nn.functional as F

# 目前版本：含patch专家交互
class LinearMlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchCoreInteraction(nn.Module):
    def __init__(self, hidden_size, core_num, head=4, drop=0.1):
        super().__init__()
        if core_num <= 0:
            raise ValueError("core_num must be positive")
        if head <= 0:
            raise ValueError("head must be positive")
        if hidden_size % head != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by head ({head})")

        self.head_dim = hidden_size // head
        self.hidden_size = hidden_size
        self.core_num = core_num # 专家数量
        self.head = head

        self.cores = nn.Parameter(torch.randn((head, core_num, self.head_dim)))# 使用可学习的核心向量作为专家进行信息交互
        self.value = nn.Linear(hidden_size, hidden_size)

        ffn_hidden_dim = 4 * (hidden_size + hidden_size) 
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(ffn_hidden_dim, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_data):
        b, p, d = input_data.shape

        q = self.value(input_data) # 将输入映射到查询空间，生成查询向量

        q = q.view(b, p, self.head, self.head_dim).transpose(1, 2)

        affiliation = torch.einsum('hcd, bhpd -> bhcp', self.cores, q) / (self.head_dim ** 0.5) # 计算每个patch与每个专家核心的关联度

        affiliation_core_to_patch = torch.softmax(affiliation, dim=-1) # 每个专家关注哪些patch
        affiliation_patch_to_core = torch.softmax(affiliation, dim=-2) # 每个patch关注哪些专家

        v = input_data.view(b, p, self.head, self.head_dim).transpose(1, 2)
        v_core = torch.einsum('bhpd, bhcp -> bhcd', v, affiliation_core_to_patch) # 从patch到专家的信息聚合
        v_patch = torch.einsum('bhcd, bhcp -> bhpd', v_core, affiliation_patch_to_core) # 从专家到patch的信息传播
        v = v_patch.transpose(1, 2).reshape(b, p, d)

        ffn_input = torch.cat([input_data - v, v], dim=2)
        
        ffn_output = self.ffn(ffn_input) # 前馈网络

        output = input_data + ffn_output
        output = self.norm(output)
        
        return output

class HLI(nn.Module):
    #def __init__(self, hidden_size, num, size, core_num, head=4, mlp_ratio=4.0):
    def __init__(self, hidden_size, size, core_num, head=4, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        #self.num, self.size = num, size
        self.size = size

        self.inter_patch_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # patch间交互替换为专家交互
        self.inter_patch_interaction = PatchCoreInteraction(
            hidden_size, 
            core_num=core_num, 
            head=head, 
            drop=0.1
        )
        self.inter_patch_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.inter_patch_mlp = LinearMlp(hidden_size, mlp_hidden_dim)

        self.intra_patch_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.intra_patch_interaction = LinearMlp(hidden_size, hidden_size)
        self.intra_patch_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.intra_patch_mlp = LinearMlp(hidden_size, mlp_hidden_dim)


    def forward(self, x):
        B, T, L, D = x.shape
        N = self.size
        if L == 0:
            return x
        P = L // N
        #####
        x = x.reshape(B, T, P, N, D)

        x_inter_patch = x.transpose(2, 3).reshape(B*T*N, P, D) 
        x_inter_patch_norm = self.inter_patch_norm(x_inter_patch)
        inter_patch_out = self.inter_patch_interaction(x_inter_patch_norm)

        x = x + inter_patch_out.reshape(B, T, N, P, D).transpose(2, 3)
        x = x + self.inter_patch_mlp(self.inter_patch_norm2(x))

        x_intra_patch = x.reshape(B*T*P, N, D) 
        x_intra_patch_norm = self.intra_patch_norm(x_intra_patch)
        intra_patch_out = self.intra_patch_interaction(x_intra_patch_norm)  
        x = x + intra_patch_out.reshape(B, T, P, N, D) 
        x = x + self.intra_patch_mlp(self.intra_patch_norm2(x))

        return x.reshape(B, T, -1, D)

class SqLinear(nn.Module):
    def __init__(self, tem_patchsize, tem_patchnum,
                      spa_patchsize, spa_patchnum, 
                      tod, dow,
                      layers,
                      input_dims, node_dims, tod_dims, dow_dims,
                      core_num=16, 
                      head=4,      
                      mlp_ratio=1.0
                 ):
        super(SqLinear, self).__init__()
        self.tod, self.dow = tod, dow
        self.node_dims = node_dims 

        base_dims = input_dims + tod_dims + dow_dims
        dims = base_dims

        self.input_st_fc = nn.Conv2d(in_channels=3, out_channels=input_dims, 
                                     kernel_size=(1, tem_patchsize), stride=(1, tem_patchsize), bias=True)

        self.time_in_day_emb = nn.Parameter(torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.spa_encoder = nn.ModuleList([
            HLI(dims, spa_patchsize, core_num=core_num, head=head, mlp_ratio=mlp_ratio) 
            for _ in range(layers)
        ])

        self.regression_conv = nn.Conv2d(in_channels=tem_patchnum*dims, 
                                         out_channels=tem_patchsize*tem_patchnum, 
                                         kernel_size=(1, 1), bias=True)

    def forward(self, x, te, ori_parts_idx, reo_parts_idx, reo_all_idx):
        
        B, T, N, _ = x.shape

        embeded_x = self.embedding(x, te)
        rex = embeded_x[:,:,reo_all_idx,:]

        for block in self.spa_encoder:
            rex = block(rex)

        orginal = torch.zeros(rex.shape[0], rex.shape[1], N, rex.shape[-1]).to(x.device)
        orginal[:,:,ori_parts_idx,:] = rex[:,:,reo_parts_idx,:]

        pred_y = self.regression_conv(orginal.transpose(2,3).reshape(orginal.shape[0],-1,orginal.shape[-2],1))

        return pred_y

    def embedding(self, x, te):
        b, t, n, _ = x.shape

        x1 = torch.cat([x, (te[...,0:1]/self.tod), (te[...,1:2]/self.dow)], -1).float()
        input_data = self.input_st_fc(x1.transpose(1,3)).transpose(1,3)
        t, d = input_data.shape[1], input_data.shape[-1]        

        t_i_d_data = te[:, -input_data.shape[1]:, :, 0]
        input_data = torch.cat([input_data, self.time_in_day_emb[(t_i_d_data).type(torch.LongTensor)]], -1)

        d_i_w_data = te[:, -input_data.shape[1]:, :, 1]
        input_data = torch.cat([input_data, self.day_in_week_emb[(d_i_w_data).type(torch.LongTensor)]], -1)

        return input_data
    
    def get_n_param(self):
        n_param = 0
        for param in self.parameters():
            if param.requires_grad:
                n_param += torch.numel(param)
        return n_param

# 消融实验：无patch专家交互

'''class LinearMlp(nn.Module):
    """Simplified MLP for linear block"""
    def __init__(self, in_features, hidden_features, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class HLI(nn.Module):
    """Hierarchical Linear Interaction Block (Ablation: No Patch-Core Interaction)"""
    def __init__(self, hidden_size, num, size, core_num=16, head=4, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size  # spa_patchnum, spa_patchsize
        
        # Inter-patch interaction: 只用简单的 LinearMlp 替代原来的 PatchCoreInteraction
        self.inter_patch_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.inter_patch_interaction = LinearMlp(hidden_size, hidden_size)  # 简单MLP作为inter-patch
        self.inter_patch_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.inter_patch_mlp = LinearMlp(hidden_size, mlp_hidden_dim)
        
        # Intra-patch interaction: 保持不变
        self.intra_patch_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.intra_patch_interaction = LinearMlp(hidden_size, hidden_size)
        self.intra_patch_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.intra_patch_mlp = LinearMlp(hidden_size, mlp_hidden_dim)
        
        # 注意：core_num 和 head 参数被保留但不再使用（兼容第一份代码接口）

    def forward(self, x):
        B, T, L, D = x.shape
        N = self.size
        if L == 0:
            return x
        if L % N != 0:
            raise ValueError(f"HLI FWD Error: L={L}, N={self.size}")
        P = L // N
        
        x = x.reshape(B, T, P, N, D)
        
        # Inter-patch interaction: [B, T, P, N, D] -> [B*T*N, P, D]
        x_inter_patch = x.transpose(2, 3).reshape(B*T*N, P, D)
        x_inter_patch_norm = self.inter_patch_norm(x_inter_patch)
        inter_patch_out = self.inter_patch_interaction(x_inter_patch_norm)  # 只用MLP
        
        # Reshape back
        x = x + inter_patch_out.reshape(B, T, N, P, D).transpose(2, 3)
        x = x + self.inter_patch_mlp(self.inter_patch_norm2(x))
        
        # Intra-patch interaction: [B, T, P, N, D] -> [B*T*P, N, D]
        x_intra_patch = x.reshape(B*T*P, N, D)
        x_intra_patch_norm = self.intra_patch_norm(x_intra_patch)
        intra_patch_out = self.intra_patch_interaction(x_intra_patch_norm)
        x = x + intra_patch_out.reshape(B, T, P, N, D)
        x = x + self.intra_patch_mlp(self.intra_patch_norm2(x))
        
        return x.reshape(B, T, -1, D)

class SqLinear(nn.Module):
    def __init__(self, tem_patchsize, tem_patchnum,
                 spa_patchsize, spa_patchnum, # node_num removed, dynamic
                 tod, dow,
                 layers,
                 input_dims, node_dims, tod_dims, dow_dims,
                 core_num=16,      # 保留参数，但HLI内部不再使用
                 head=4,           # 保留参数，但HLI内部不再使用
                 mlp_ratio=1.0
                 ):
        super(SqLinear, self).__init__()
        self.tod, self.dow = tod, dow
        self.node_dims = node_dims
        
        base_dims = input_dims + tod_dims + dow_dims
        dims = base_dims
        
        self.input_st_fc = nn.Conv2d(in_channels=3, out_channels=input_dims,
                                     kernel_size=(1, tem_patchsize), stride=(1, tem_patchsize), bias=True)
        
        self.time_in_day_emb = nn.Parameter(torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)
        
        # 使用修改后的 HLI（无Patch-Core交互）
        self.spa_encoder = nn.ModuleList([
            HLI(dims, spa_patchnum, spa_patchsize, core_num=core_num, head=head, mlp_ratio=mlp_ratio)
            for _ in range(layers)
        ])
        
        self.regression_conv = nn.Conv2d(in_channels=tem_patchnum*dims,
                                         out_channels=tem_patchsize*tem_patchnum,
                                         kernel_size=(1, 1), bias=True)
    
    def forward(self, x, te, ori_parts_idx, reo_parts_idx, reo_all_idx):
        """
        Forward pass with dynamic node handling
        
        Args:
            x: [B, T, N, 1] input traffic (N can vary)
            te: [B, T, N, 2] time information
            ori_parts_idx: original parts indices
            reo_parts_idx: reordered parts indices
            reo_all_idx: all reordered indices
        """
        # x: [B,T,N,1] input traffic
        # te: [B,T,N,2] time information
        B, T, N, _ = x.shape  # Dynamic node number from input

        # Spatio-temporal embedding (node-agnostic)
        embeded_x = self.embedding(x, te)
        rex = embeded_x[:,:,reo_all_idx,:]  # select patched points

        # Hierarchical linear interaction encoding
        for block in self.spa_encoder:
            rex = block(rex)

        # Restore original spatial structure
        orginal = torch.zeros(rex.shape[0], rex.shape[1], N, rex.shape[-1]).to(x.device)
        orginal[:,:,ori_parts_idx,:] = rex[:,:,reo_parts_idx,:]  # back to the original indices

        # Projection decoding
        pred_y = self.regression_conv(orginal.transpose(2,3).reshape(orginal.shape[0],-1,orginal.shape[-2],1))

        return pred_y  # [B,T,N,1]

    def embedding(self, x, te):
        """
        Node-agnostic embedding function (like STOP's MLP)
        - No node-specific embedding matrix
        - Uses node-agnostic spatial projection
        - Can handle variable number of nodes
        """
        b, t, n, _ = x.shape

        # Combine input traffic with time features
        x1 = torch.cat([x, (te[...,0:1]/self.tod), (te[...,1:2]/self.dow)], -1).float()
        input_data = self.input_st_fc(x1.transpose(1,3)).transpose(1,3)
        t, d = input_data.shape[1], input_data.shape[-1]        

        # Add time of day embedding (node-agnostic, like STOP)
        t_i_d_data = te[:, -input_data.shape[1]:, :, 0]
        input_data = torch.cat([input_data, self.time_in_day_emb[(t_i_d_data).type(torch.LongTensor)]], -1)

        # Add day of week embedding (node-agnostic, like STOP)
        d_i_w_data = te[:, -input_data.shape[1]:, :, 1]
        input_data = torch.cat([input_data, self.day_in_week_emb[(d_i_w_data).type(torch.LongTensor)]], -1)

        return input_data
    
    def get_n_param(self):
        """Count trainable parameters"""
        n_param = 0
        for param in self.parameters():
            if param.requires_grad:
                n_param += torch.numel(param)
        return n_param'''