import torch
import torch.nn as nn

class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        # 每个输入特征一组多项式系数
        self.poly_coeff = nn.Parameter(torch.randn(input_dim, degree + 1))
        # 线性组合权重
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: (batch, input_dim)
        exponents = torch.arange(self.degree + 1, device=x.device)
        x_expanded = x.unsqueeze(-1) ** exponents  # (batch, input_dim, degree+1)
        x_nonlinear = torch.sum(x_expanded * self.poly_coeff, dim=-1)  # (batch, input_dim)
        output = self.linear(x_nonlinear)  # (batch, output_dim)
        return output

class KAN(nn.Module):
    def __init__(self, input_dim=12, output_dim=1, degree=3):
        super().__init__()
        self.kan_layer = KANLayer(input_dim, output_dim, degree)
        self.res = nn.Linear(input_dim, output_dim)

    def forward(self, x, occ=None):
        # x: (batch, seq_len, input_dim)
        res = self.res(x)
        batch_size, seq_len, _ = x.shape
        x = x.view(-1, self.kan_layer.input_dim)  # (batch*seq_len, input_dim)
        output = self.kan_layer(x)  # (batch*seq_len, output_dim)
        return output.view(batch_size, seq_len, -1) + res  # (batch, seq_len, output_dim)