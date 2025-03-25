import torch
import torch.nn as nn
d_mode = 512

ffn_multi_head = nn.Linear(d_mode, d_mode)
dropout_multi_head = nn.Dropout(0.1)

def multihead_att(q:torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, head_num:int=8):
    """
    多头注意力：01 变形 最后两个维度 第二个维度为head 02 mask 形状预设 03 最后fusion的linear和dropout
    """
    bsz, sq_length, d_mode = q.shape
    head_dim = d_mode // head_num

    q = q.reshape(bsz, sq_length, head_num, head_dim).transpose(1, 2) # trans to: bsz, head_num, sq_length, head_dim
    k = k.reshape(bsz, sq_length, head_num, head_dim).transpose(1, 2)
    v = v.reshape(bsz, sq_length, head_num, head_dim).transpose(1, 2)
    # mask shape (bsz, 1, sq_length), filled with 1 and -inf
    mask = mask.reshape(bsz, 1, 1, sq_length)

    scale = torch.sqrt(torch.tensor(head_dim))
    atten = torch.matmul(q, k.transpose(3, 2)) / scale
    atten = atten * mask # 1, -inf 
    atten = torch.softmax(atten, dim=-1)
    oput = torch.matmul(atten, v)
    
    oput = oput.transpose(1, 2).reshape(bsz, sq_length, -1)    
    oput = ffn_multi_head(oput)
    oput = dropout_multi_head(oput)
    return oput

# mask shape(bsz, sqlen)

if __name__ == '__main__':
    q = torch.ones((2, 256, 512))
    k = torch.ones((2, 256, 512))
    v = torch.ones((2, 256, 512))
    mask = torch.ones((2, 1, 256))
    mask[0, 0, 0] = -torch.inf
    output = multihead_att(q, k, v, mask) 
    print(output.shape)









    