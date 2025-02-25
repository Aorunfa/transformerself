"""
ChatGLM对ROPE论文的代码实现
"""

import torch
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
         # 计算 \theta_i
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq.half()
        
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            # 对应m * \theta
            emb = torch.einsum('i,j->ij', t, self.inv_freq)   # 计算外积
            if self.precision == torch.bfloat16:
                emb = emb.float()
            
            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()  # 计算得到cos(m*\theta)
            sin_cached = emb.sin()  # 计算得到sin(m*\theta)
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def _apply(self, fn):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)

def apply_rotary_emb(
    x: torch.Tensor,
    cos_theta: torch.Tensor,
    sin_theta: torch.Tensor,
) -> torch.Tensor:              # 通过逐元素乘法实现
    x_rot = torch.cat([x[..., ::2] * cos_theta - x[..., 1::2] * sin_theta,
                    x[..., ::2] * sin_theta + x[..., 1::2] * cos_theta], dim=-1)
    return x_rot # 注意 这里的元素顺序[0, 2, 4, ..., 1, 3, 5, ...], 但在q*k^T计算中由于加性原理不影响最终的得分

if __name__ == '__main__':
    Rope = RotaryEmbedding(dim=1024)
    x = torch.randn((32, 512, 1024))
    cos_theta, sin_theta = Rope(x)  # get sin cos, [m * theta0, m * theta1, m * theta2, ...] 
    print(cos_theta.shape)
    x = apply_rotary_emb(x, cos_theta, sin_theta)
    print(x.shape)
    print(x.is_contiguous())


