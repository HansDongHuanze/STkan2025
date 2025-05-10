import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward

class WaveletGraph1DNonLinear(nn.Module):
    """
    对每个特征单独做小波+图attention一维非线性
    输入: (batch, node, dim)
    输出: (batch, node, dim)
    """
    def __init__(self, dim, wavelet_type='db1', dwt_level=1, adj=None):
        super().__init__()
        self.dim = dim
        self.wavelet_type = wavelet_type
        self.dwt_level = dwt_level
        self.dwt = DWT1DForward(J=dwt_level, wave=wavelet_type, mode='zero')
        # 线性层输入维度需根据小波变换后长度动态调整
        test_arr = torch.zeros(1, 1, 1)
        cA, _ = self.dwt(test_arr)
        self.wave_len = cA.shape[-1]
        self.linear = nn.ModuleList([nn.Linear(self.wave_len, 1) for _ in range(dim)])
        self.attn_fc = nn.ModuleList([nn.Linear(1, 1) for _ in range(dim)])
        self.adj = adj

    def forward(self, x):
        # x: (batch, node, dim)
        batch, node, dim = x.shape
        outs = []
        for i in range(dim):
            xi = x[:, :, i]  # (batch, node)
            xi_flat = xi.contiguous().view(-1, 1)  # (batch*node, 1)
            # 小波特征
            xi_wave = self._wavelet_feature(xi_flat)  # (batch*node, 1)
            xi_wave = xi_wave.view(batch, node, 1)  # (batch, node, 1)
            # 图attention（对每个特征单独）
            attn_scores = self.attn_fc[i](xi_wave).squeeze(-1)  # (batch, node)
            adj_mask = (self.adj > 0).float()  # (node, node)
            attn_matrix = attn_scores.unsqueeze(1).repeat(1, node, 1)  # (batch, node, node)
            attn_matrix = attn_matrix * adj_mask.unsqueeze(0)
            attn_matrix = torch.softmax(attn_matrix.masked_fill(adj_mask.unsqueeze(0)==0, -1e9), dim=-1)
            xi_out = torch.einsum('bij,bjd->bid', attn_matrix, xi_wave)  # (batch, node, 1)
            outs.append(xi_out)
        out = torch.cat(outs, dim=-1)  # (batch, node, dim)
        return out

    def _wavelet_feature(self, x):
        # x: (batch*node, 1)
        batch = x.shape[0]
        x_reshape = x.view(batch, 1, 1)
        cA, _ = self.dwt(x_reshape)
        cA = cA.squeeze(1)  # (batch, wave_len)
        out = self.linear[0](cA)  # (batch, 1)
        out = F.softplus(out)
        return out

class WGAKLayer(nn.Module):
    def __init__(self, in_dim, out_dim, wavelet_type='db1', dwt_level=1, adj=None):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.nonlinear = WaveletGraph1DNonLinear(out_dim, wavelet_type, dwt_level, adj)
        self.linear2 = nn.Linear(out_dim, out_dim)
    def forward(self, x):
        # x: (batch, node, in_dim)
        x = self.linear1(x)
        x = self.nonlinear(x)
        x = self.linear2(x)
        return x

class WGAK(nn.Module):
    def __init__(self, input_dim=12, output_dim=1, hidden_dim=16, num_layers=2, wavelet_type='db1', dwt_level=1, adj=None):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(WGAKLayer(in_dim, out_dim, wavelet_type, dwt_level, adj))
            in_dim = out_dim
        self.layers = nn.ModuleList(layers)
        self.res = nn.Linear(input_dim, output_dim)
    def forward(self, x, occ=None):
        # x: (batch, seq_len, input_dim)
        res = self.res(x)
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, input_dim)
        for layer in self.layers:
            x = layer(x)
        return x + res  # (batch, seq_len, output_dim)