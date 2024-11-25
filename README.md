# transformerself 介绍
一个用于transformer llm模型学习的库，梳理llm训练的的基本步骤，微调方法及原理， 整理能快速理解并上手的代码实战项目

# transformer 论文原理
· 首先推荐先阅读[周弈帆的博客解读transformer](https://zhouyifan.net/2022/11/12/20220925-Transformer/)， 达到能够理解以下要点  
  1. 注意力机制: ```q*K^T```做一次向量化查询，```sofmax(q*K^T / sqrt(d_model)) * V```完成查询结果的加权, sqrt(d_model)用于softmax缩放，将梯度集中在明显变化区域  
  2. 多头注意力设计: 折叠分组查询，使用更少的参数量，进行更多特征空间的交互
  3. 注意力mask: 在seq_length维度保证当前查询只能看到自己以及之前的信息，模拟rnn的串行输出
  4. 可训参数矩阵```Wq Wk Wv Wo``` 实现类似自动化特征工程的效果，如对一个查询向量q计算```q * Wk^T``` 可以得到新的查询，查询优化方向和loss下降方向一致   
  5. FFN前馈神经网络, 隐藏层维度设置```4*d_model```，特征在更高维的隐空间交互，实现类似特征增强的效果, 4这个值目前看是约定俗成，没太多意义    
  6. pos embeding沿着seq_length和d_model对应的两个维度add pos数值，标记位置信息。简单理解一个特征矩阵Q中任意一个数值通过向前diff和向上diff可以得到位置坐标，模型可以学到这种模式   
  7. token_embedding矩阵为可学习矩阵，实现将一个token_id转换为对应embedding向量，维度为d_model
  8. 训练阶段，对比rnn, transformer能够做到训练的并行，即```one sentence -> one output -> one loss```， 得力于注意力mask的设计
  9. 预测阶段，于rnn相同，transformer自回归预测下一个token，当出现终止符则停止
· 论文链接[attendion all you need](https://arxiv.org/abs/1706.03762)，论文模型结构为encoder-decoder的结构

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
