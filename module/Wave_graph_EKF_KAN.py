import torch
import torch.nn as nn

class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size=5, degree=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.degree = degree
        
        # 可学习的样条系数 (每个输入特征对应一组系数)
        self.spline_coeff = nn.Parameter(torch.randn(input_dim, grid_size + degree))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        # x形状: (batch_size, input_dim)
        # 生成多项式基函数展开（简化版示例）
        x_expanded = x.unsqueeze(-1)  # (batch, input_dim, 1)
        exponents = torch.arange(self.grid_size + self.degree, device=x.device)
        x_poly = x_expanded ** exponents  # (batch, input_dim, grid+degree)
        
        # 计算样条输出
        output = torch.einsum('bid,id->b', x_poly, self.spline_coeff)
        return output.unsqueeze(-1) + self.bias

class KAN(nn.Module):
    def __init__(self, input_dim=12, output_dim=1):
        super().__init__()
        self.kan_layer = KANLayer(input_dim, output_dim)
        
    def forward(self, x, occ):
        # 输入形状: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        x = x.view(-1, self.kan_layer.input_dim)  # 展平批次和序列维度
        output = self.kan_layer(x)
        return output.view(batch_size, seq_len, -1)  # 恢复原始形状