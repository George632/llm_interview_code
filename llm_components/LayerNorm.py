import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps =1e-6):
        super().__init__()
        
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        #求均值和方差
        mean = torch.mean(x, dim=-1, keepdim=True) #[b, s, 1]
        var= torch.var(x, dim=-1, keepdim=True, unbiased=False)
        #正则化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        #加偏移
        return self.gamma * x_normalized + self.beta
