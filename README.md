# transformerself 介绍
一个用于transformer llm模型学习的库，梳理llm训练的的基本步骤，微调方法及原理， 分享能快速理解并上手的代码实战项目

# transformer 论文原理
首先推荐先阅读[周弈帆的博客解读transformer](https://zhouyifan.net/2022/11/12/20220925-Transformer/)， 达到能够理解以下要点  
  1. 注意力机制: ```q*K^T```做一次向量化查询，```sofmax(q*K^T / sqrt(d_model)) * V```完成查询结果的加权, sqrt(d_model)用于softmax缩放，将梯度集中在明显变化区域  
  2. 多头注意力设计: 折叠分组查询，使用更少的参数量，进行更多特征空间的交互
  3. 注意力mask: 在seq_length维度保证当前查询只能看到自己以及之前的信息，模拟rnn的串行输出
  4. 可训参数矩阵```Wq Wk Wv Wo``` 实现类似自动化特征工程的效果，如对一个查询向量q计算```q * Wk^T``` 可以得到新的查询，查询优化方向和loss下降方向一致   
  5. FFN前馈神经网络, 隐藏层维度设置```4*d_model```，特征在更高维的隐空间交互，实现类似特征增强的效果, 4这个值目前看是约定俗成，没太多意义    
  6. pos embeding沿着seq_length和d_model对应的两个维度add pos数值，标记位置信息。简单理解一个特征矩阵Q中任意一个数值通过向前diff和向上diff可以得到位置坐标，模型可以学到这种模式   
  7. token_embedding矩阵为可学习矩阵，实现将一个token_id转换为对应embedding向量，维度为d_model
  8. 训练阶段，对比rnn, transformer能够做到训练的并行，即```one sentence -> one output -> one loss```， 得力于注意力mask的设计
  9. 预测阶段，与rnn相同，transformer自回归预测下一个token，当出现终止符则停止  
论文链接[attendion all you need](https://arxiv.org/abs/1706.03762)，论文模型结构为encoder-decoder的结构  

# transformer 论文代码解读
  首先根据[周弈帆的博客-PyTorch Transformer 英中翻译超详细教程](https://zhouyifan.net/2023/06/11/20221106-transformer-pytorch/)手撕一遍transformer的代码，了解各个组件设计以及这类代码设计风格。该代码基本与transformer论文结构相同，唯一的区别在于最后的ouput head是一个单独的线性层，与embeding层不共享权重

# llm模型训练流程及方法
  推荐根据轻量化llm项目完整走一遍对话llm模型的开发[Minimind](https://github.com/jingyaogong/minimind), 只要求跑通进行代码阅读的情况下，4Gb显存的卡将batchsize设置为1可以吃得消   
## 一. pretrained
  01 prtrained的目的是让模型具备合理预测下一个token的能力，合理体现在能够根据一个字输出符合逻辑的话一段话，简而言之就是字接龙  
  02 prtrained的输入是```input_ids[:-1]```， 标签是```input_ids[1:]```，input_ids是指文字经过tokenize后的idlist，如```我爱你 --> <s>我爱你<\s> --> [1, 2, 23, 4, 2]```，之所以输入与标签要错一位，目的在于实现预测下一个token的监督学习，例如输入文字是”我爱你啊“， 那么预测下一个token逻辑是```我 --> 我爱; 我爱 --> 我爱你；我爱你 --> 我爱你啊```， 使用mask能对信息进遮掩，实现并行训练，即模型ouput中的每一个位置是由该位置之前的所有信息预测得到的, 初始的```ouput[0]```则由```<s>我```预测得到  
  03 prtrained的损失函数为corss_entropy，模型输出的logits维度为(batch_size, max_seq_length, voc_size), max_seq_length为文字对齐到的最大长度，voc_size为词表的token数量。损失计算的逻辑为对logits沿最后的轴进行softmax得到几率，形状不变；沿着max_seq_length取出label对应的token_id计算corss_entropy；由于label的真实长度不一定为max_seq_length，需要设置一个真实token_id的掩码就行过滤  
  
## 二. sft
  sft监督微调的目的是让模型具备对话能力，通过将prompt嵌入问答模版，如```用户<s>说:你是谁？</s>\n助手<s>回答:我是人工智能助手</s>\n```，构成一个新的语料微调pretrained模型。  
  对话模板是为了引入特殊的字符，通过微调能够让模型理解问题句柄，从而预测问题后面的答案。  
  sft与prtrained区别在于损失的计算以及训练的参数。sft只计算output中对应标签回答的部分，其余部分不计入损失，但这些部分会在attention中被关注到；训练参数取决于不同的微调方法，常见：full-sft, lora, bitfit, preEmbed, prefix, adapter等
### 01 full-sft 全量微调
  全量微调是指使用pretrained初始化权重，对模型的全部参数进行训练，语料设计和损失设计同上  
### 02 lora-sft 低秩矩阵自适应微调
  [lora](https://arxiv.org/abs/2106.09685)对可学习矩阵W（Wq Wk Wv Wo ...）增加两个低秩矩阵A和B，对输入进行矩阵乘法并相加``` XW + XAB = X(W + AB) = XW` ```，``` W` ```为更新后的参数矩阵。假设W的维度为```(d_k, d_model)```, AB维度应该满足```(dk, r) (r, d_model)```，r为秩参数，r越大AB参数越多，W可更新的```△W```分布自由度更大。  
  相比全量微调lora需要的显存大大减小，但在小模型上训练速度不一定更快（小模型forward过程耗时占比大） 
### 03 其他微调方法
  · PreEmbed，只微调token embedding参数矩阵，适应新的数据分布
  · Prompt-tuning 在输入token前增加特殊的提示token，只微调提示token的embeding向量参数，适合小模型适配下游任务
  · P-tunning 是Prompt tuning的进阶版，提示token可插入prompt的指定位置
  · Prefix，在attention中```K=XWk V=XWv```对X增加可学习前缀token embeding矩阵，作为虚拟的提示上下文, ```K=[P; X]Wk V=[P; X]Wv```P是可学习的参数矩阵，维度(L, d_model)，L表示需要增加的提示前缀长度，是超参数。```[P; X]```表示在X输入矩阵开始位置拼接矩阵P。prefix微调的是每一个transform层中的attention可学习前缀矩阵P，不同的层中，P不同。    
  · Bitfit: 只微模型的偏置项，偏置项出现在所有线性层和Layernorma层中。    
  · Adapter，在transform模块的多头注意力与输出层之后增加一个adpter层，只微调adpter参数。 adpter包含```下投影linear + nolinear + 上投影linear; skip-connect结构```， 中间结构类似lora变现为nonlinear(XA)B的结构，skip-connect结构保证的模型能力最多退化为原模型；由于改变了Laynorm输入的数据分布，Laynorm的scale参数也需要加入训练。  

## 三. preference opimized
  偏好对齐(优化)的目的是让模型的输出更加符合用户的习惯，包括文字逻辑、风格、伦理性、安全性等  
### 01 rlhf
  pending 需要梳理强化学习的基础理论才能进阶
### 02 dpo
  直接偏好优化(direct-preference-opimized)与rlhf不同，直接跳过了奖励模型的训练，根据偏好数据一步到位训练得到对齐模型。[论文](https://arxiv.org/abs/2305.18290)解读可以参考博客[人人都能看懂的DPO数学原理](https://mp.weixin.qq.com/s/aG-5xTwSzvHXN4B73mfKMA)  
  筒体而言，dpo从rlhf总体优化目标出发```模型输出尽可能接近偏好标签，尽可能偏离非偏好标签，尽可能少偏离原模型输出```，推导最优奖励模型的显示解，代入奖励模型的损失函数，得到一个只与待训模型有关的损失函数，该函数就是偏好优化的目标。 
  手撕dpo训练代码可以参考，有助于快速理解dpo损失的计算过程```/minimind/5-dpo_train_self.py```
## 四. evalization

## 五. RNN补充
...

# appendix
## 01 Model structure
### RMSNorme VS LayerNorme VS batchNorm
  操作解析 + 优势 
### sin-cos pos embedding vs ROPE
  操作解析 + 优劣
### linear attention
  解决的问题，组件设计
## 02 Large model fine tune
### Q-Lora
  应用场景 双重量化 + lora微调，牺牲计算的效率换内存
## 03 quantilization
  常见的量化方法有哪些
