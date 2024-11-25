# transformerself 介绍
一个用于transformer llm模型学习的库，梳理llm训练的的基本步骤，微调方法及原理， 整理能快速理解并上手的代码实战项目

# transformer 论文原理
· 首先推荐先阅读[周弈帆的博客解读transformer](https://zhouyifan.net/2022/11/12/20220925-Transformer/)， 达到能够理解以下要点
  · 注意力机制，q, k完成向量化查询，sorfmax(q * k^T / sqrt(d_model)) * v 完成查询结果的加权, sqrt(d_model)用于softmax缩放，梯度集中在明显变化区域  
  · 多头注意力设计，使用更少的参数量，达到更多的特征空间的交互，效果更好  
  · 可训参数矩阵Wq Wk Wv Wo 实现类似自动化特征工程的效果，如一个查询 q * Wk^T 得到新的查询，该查询优化方向和loss下降方向一致   
  · FFN前馈神经网络, 隐藏层维度设置4*d_model，特征在更高维的隐空间交互，实现类似特征增强的  
  · position embeding沿着seq_length和d_model对应的两个维度add数值，标记位置信息； 简单理解一个特征矩阵Q中任意一个数值通过向前diff和向上diff可以得到位置坐标，模型可以学到这个patten  
  · token embedding矩阵实现一个token_id转换为一个embedding  
  · 对比rnn, transformer能够做到训练的并行，one sentence -> one output -> one loss  
· 论文链接[attendion all you need](https://arxiv.org/abs/1706.03762)
# transformer 论文代码解读

# llm模型训练流程及方法
## 01 pretrained
## 02 sft
### full sft
### lora sft
### otrers sft
## 03 preference opimized
### dpo
### rlhf
## 04 evalization
...

# appendix
## 01 Model structure
### RMSNorme VS LayerNorme
### sin-cos posistion enbeddinig vs ROPE
### linear attention
## 02 Finetune
### Q-Lora
## 03 quantilization
## 04 others
...
