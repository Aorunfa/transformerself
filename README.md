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
  9. 预测阶段，与rnn相同，transformer自回归预测下一个token，当出现终止符则停止
· 论文链接[attendion all you need](https://arxiv.org/abs/1706.03762)，论文模型结构为encoder-decoder的结构

# transformer 论文代码解读
  首先根据[周弈帆的博客-PyTorch Transformer 英中翻译超详细教程](https://zhouyifan.net/2023/06/11/20221106-transformer-pytorch/)手撕一遍transformer的代码，了解各个组件设计以及这类代码设计风格。该代码基本与transformer论文结构相同，唯一的区别在于最后的ouput head是一个单独的线性层，与embeding层不共享权重

# llm模型训练流程及方法
  推荐根据轻量化llm项目完整走一遍对话llm模型的开发[Minimind](https://github.com/jingyaogong/minimind), 只要求跑通进行代码阅读的情况下，8Gb显存的卡将batchsize设置为1基本能吃得消   
## 01 pretrained
  · prtrained的目的是让模型具备合理预测下一个token的能力，合理体现在能够根据一个字输出符合逻辑的话一段话，简而言之就是字接龙  
  · prtrained的输入是input_ids[:-1]， 标签是input_ids[1:]，input_ids是指文字经过tokenize后的idlist，如```我爱你 --> <s>我爱你<\s> --> [1, 2, 23, 4, 2]```，之所以输入与标签要错一位，目的在于实现预测下一个token的监督学习，例如输入文字是”我爱你啊“， 那么预测下一个token逻辑是```我 --> 我爱; 我爱 --> 我爱你；我爱你 --> 我爱你啊```， 使用mask能对信息进遮掩，实现并行训练，即模型ouput中的每一个位置是由该位置之前的所有信息预测得到的, 初始的```ouput[0]```则由```<s>我```得到  
  · prtrained的损失函数为corss_entropy，模型输出的logits维度为(batch_size, max_seq_length, voc_size), max_seq_length为文字对齐到的最大长度，voc_size为词表的token数量。损失计算的逻辑为对logits沿最后的轴进行softmax得到几率，形状不变；沿着max_seq_length取出label对应的token_id计算corss_entropy；由于label的真实长度不一定为max_seq_length，需要设置一个真实token_id的掩码就行过滤
## 02 sft
  sft监督微调的目的是让模型具备对话能力，通过特殊的问答token模板然触发模型的回答;
  sft监督微调输出需要经过对话模板构造，
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
