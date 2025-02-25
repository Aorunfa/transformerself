"""
Llama ROPE实现
"""
import torch
from typing import Tuple
import torch.nn.functional as F
import torch.nn as nn
import math

# 生成旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # 1000 ** (-2i/dim), i=[0, 1,..., d/2 -1]
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # (模长, 角度), freqs shape(seq_length, dim//2), freqs [cos(O) + i*sin(O)], O = seq_idx*freqs[dim_idx // 2]
    return freqs_cis

# 旋转位置编码计算
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)     # 先执行复数乘法，复数转化为二维向量，展平回原来的形状
    return xq_out.type_as(xq), xk_out.type_as(xk)               # 转换为输入的数值类型


class SelfAttention(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()

        self.wq = nn.Linear(dim, dim, bias=True)
        self.wk = nn.Linear(dim, dim, bias=True)
        self.wv = nn.Linear(dim, dim, bias=True)        
        self.freqs_cis = precompute_freqs_cis(dim, max_seq_len)
        self.dim = dim

    def forward(self, x: torch.Tensor):
        # batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # xq = xq.view(batch_size, seq_len, dim)
        # xk = xk.view(batch_size, seq_len, dim)
        # xv = xv.view(batch_size, seq_len, dim)

        # attention 操作之前，应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=self.freqs_cis)
        
        # scores.shape = (bs, seqlen, seqlen)
        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(self.dim)
        scores = F.softmax(scores.float(), dim=-1)
        output = torch.matmul(scores, xv)           # (batch_size, seq_len, dim)
        return output

if __name__ == '__main__':
    x = torch.randn((1, 512, 1024))
    atten = SelfAttention(dim=1024, max_seq_len=512)
    x = atten(x)


