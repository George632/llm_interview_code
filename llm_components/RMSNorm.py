import torch
from torch import nn
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size)
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x):
       	output = self._norm(x.float()).type_as(x)
        return output * self.weight
    #打印字符串
    def extra_repr(self):
        return f"{tuple(self.weight.shape), eps={self.eps}}"
