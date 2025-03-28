# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MY_INF = 1e12


class PositionalEncoding(nn.Module):
    """
    position for seq_idx and d_model_idx
    """
    def __init__(self, d_model: int, max_seq_len: int):
        """
        d_model: token embdeing length
        max_seq_len: max seq token num
        PE(pos, 2i) = sin(pos/10000**(2i/dmodel))
        PE(pos, 2i + 1) = cos(pos/10000**(2i/dmodel))
        """
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len) # for seq pos
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)    # for embd pos, 2i
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(1, max_seq_len, d_model) # 1 for batch; d_model: 2i , 2i + 1, ...

        self.register_buffer('pe', pe, False) # save for catch 

    def forward(self, x: torch.Tensor):
        n, seq_len, d_model = x.shape
        pe: torch.Tensor = self.pe
        assert seq_len <= pe.shape[1]
        assert d_model == pe.shape[2]
        rescaled_x = x * d_model**0.5               # TODO
        return rescaled_x + pe[:, 0:seq_len, :]


def attention(q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              mask: Optional[torch.Tensor] = None):
    '''
    Note: The dtype of mask must be bool
    qi_lenght = ki_lenght
    k_num = q_num 
    '''
    # q shape: [n, heads, q_len, d_k]
    # k shape: [n, heads, k_len, d_k]
    # v shape: [n, heads, k_len, d_v]
    assert q.shape[-1] == k.shape[-1]
    d_k = k.shape[-1]
    # tmp shape: [n, heads, q_len, k_len]
    tmp = torch.matmul(q, k.transpose(-2, -1)) / d_k**0.5  #attention weight: QKt / d ** 0.5
    if mask is not None:
        tmp.masked_fill_(mask, -MY_INF)
    tmp = F.softmax(tmp, -1)
    tmp = torch.matmul(tmp, v)  # tmp shape: [n, heads, q_len, d_v]
    return tmp


class MultiHeadAttention(nn.Module):

    def __init__(self, heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        """
        heads: split d_model in to heads chunck
        d_model: embded dim lenght
        """

        assert d_model % heads == 0
        # dk == dv
        self.d_k = d_model // heads
        self.heads = heads
        self.d_model = d_model
        
        self.q = nn.Linear(d_model, d_model)  
        """
        note: Wq, params is learnable, Q = EWq
            E shape: [..., seq_len, d_model]
            Wq shape: [d_model, d_model]
            qi ki vi of one token should only denote by it self 
            qi of every token share one Wq
            Wq learnable params is d_model * d_model, so can use nn.Linear(d_model, d_model) to replace this process
        """

        self.k = nn.Linear(d_model, d_model)  
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)  # for all feature fusion
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        # batch should be same
        assert q.shape[0] == k.shape[0] 
        assert q.shape[0] == v.shape[0]
        # the sequence length of k and v should be aligned
        assert k.shape[1] == v.shape[1] # batch, seq_len, 

        n, q_len = q.shape[0:2]
        n, k_len = k.shape[0:2]
        q_ = self.q(q).reshape(n, q_len, self.heads, self.d_k).transpose(1, 2) # split for head, shape: batch, head, len, dk
        k_ = self.k(k).reshape(n, k_len, self.heads, self.d_k).transpose(1, 2)
        v_ = self.v(v).reshape(n, k_len, self.heads, self.d_k).transpose(1, 2)

        attention_res = attention(q_, k_, v_, mask) # n, head, q_len, d_v;  d_v=d_model // heads
        concat_res = attention_res.transpose(1, 2).reshape(
            n, q_len, self.d_model)
        concat_res = self.dropout(concat_res)

        output = self.out(concat_res)
        return output


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff) # d_ff : hidden dim 
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(F.relu(x))
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self,
                 heads: int,
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(heads, d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) # only normalize the last dim for each token
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask: Optional[torch.Tensor] = None):
        tmp = self.self_attention(x, x, x, src_mask)
        tmp = self.dropout1(tmp)
        x = self.norm1(x + tmp)     # add and norm， shape: [n, seq_len, d_model]
        tmp = self.ffn(x)           # each token fuse itself
        tmp = self.dropout2(tmp)
        x = self.norm2(x + tmp)
        return x


class DecoderLayer(nn.Module):

    def __init__(self,
                 heads: int,
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(heads, d_model, dropout)
        self.attention = MultiHeadAttention(heads, d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,
                x,
                encoder_kv: torch.Tensor,
                dst_mask: Optional[torch.Tensor] = None,
                src_dst_mask: Optional[torch.Tensor] = None):
        tmp = self.self_attention(x, x, x, dst_mask)
        tmp = self.dropout1(tmp)
        x = self.norm1(x + tmp)
        tmp = self.attention(x, encoder_kv, encoder_kv, src_dst_mask)
        tmp = self.dropout2(tmp)
        x = self.norm2(x + tmp)
        tmp = self.ffn(x)
        tmp = self.dropout3(tmp)
        x = self.norm3(x + tmp)
        return x


class Encoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 pad_idx: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 120):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx) # idx to one-hot to embeding vect, pad_idx require no grad
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.layers = []
        for i in range(n_layers):
            self.layers.append(EncoderLayer(heads, d_model, d_ff, dropout))
        self.layers = nn.ModuleList(self.layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask: Optional[torch.Tensor] = None):
        x = self.embedding(x)  # shape: batch, seq_len, d_model
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 pad_idx: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 120):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.layers = []
        for i in range(n_layers):
            self.layers.append(DecoderLayer(heads, d_model, d_ff, dropout))
        # self.layers = nn.Sequential(*self.layers)
        self.layers = nn.ModuleList(self.layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x,
                encoder_kv,
                dst_mask: Optional[torch.Tensor] = None,
                src_dst_mask: Optional[torch.Tensor] = None):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_kv, dst_mask, src_dst_mask)
        return x


class Transformer(nn.Module):

    def __init__(self,
                 src_vocab_size: int,
                 dst_vocab_size: int,
                 pad_idx: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 200):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, pad_idx, d_model, d_ff,
                               n_layers, heads, dropout, max_seq_len)
        self.decoder = Decoder(dst_vocab_size, pad_idx, d_model, d_ff,
                               n_layers, heads, dropout, max_seq_len)
        self.pad_idx = pad_idx
        self.output_layer = nn.Linear(d_model, dst_vocab_size)

    def generate_mask(self,
                      q_pad: torch.Tensor,
                      k_pad: torch.Tensor,
                      with_left_mask: bool = False):
        # q_pad shape: [n, q_len]
        # k_pad shape: [n, k_len]
        # q_pad k_pad dtype: bool
        # with_left_mask for encoder-decorder mask
        assert q_pad.device == k_pad.device
        n, q_len = q_pad.shape
        n, k_len = k_pad.shape

        mask_shape = (n, 1, q_len, k_len)
        if with_left_mask:
            mask = 1 - torch.tril(torch.ones(mask_shape)) # 上三角0, 表示注意力权重不填充
        else:
            mask = torch.zeros(mask_shape)
        mask = mask.to(q_pad.device)
        for i in range(n):
            mask[i, :, q_pad[i], :] = 1 # to fill with -MY_INF
            mask[i, :, :, k_pad[i]] = 1
        mask = mask.to(torch.bool)
        return mask

    def forward(self, x, y):
        # x, y shape: b, seq_len
        src_pad_mask = x == self.pad_idx
        dst_pad_mask = y == self.pad_idx
        src_mask = self.generate_mask(src_pad_mask, src_pad_mask, False)
        dst_mask = self.generate_mask(dst_pad_mask, dst_pad_mask, True)
        src_dst_mask = self.generate_mask(dst_pad_mask, src_pad_mask, False)
        encoder_kv = self.encoder(x, src_mask)
        res = self.decoder(y, encoder_kv, dst_mask, src_dst_mask)
        res = self.output_layer(res)
        return res

if __name__ == '__main__':
    # position
    # d_model=512
    # max_seq_len = 256
    # pe = PositionalEncoding(d_model, max_seq_len)
    # x = torch.ones((1, 120, 512))
    # x = pe(x)
    # print(x)
    # #print(x.shape)
    
    # batch, seq_len, 
    # multi
    q = torch.ones((2, 256, 512))
    k = torch.ones((2, 256, 512))
    v = torch.ones((2, 256, 512))
    mtl = MultiHeadAttention(8, 512)
    res = mtl(q, k ,v)
    print(res.shape)


    # mask_shape = (2, 1, 128, 256)
    # mask = 1 - torch.tril(torch.ones(mask_shape)) # 上三角 元素填充为1
    # a = 1
    """
    疑问: 
        00 train的encoder input和decoder input, decoder output是怎么样的
        01 inference的input和output是怎么样的
    s5cmd --credentials-file ~/.your-credentials-file --profile my-work-profile ls s3://my-company-bucket/
    export S3_ENPOINT_URL=https://storage.googleapis.com
    """