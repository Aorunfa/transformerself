# 一. 介绍

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
一个用于transformer llm模型学习的仓库，梳理llm训练的的基本步骤，微调方法及原理， 分享能快速理解并上手的代码实战项目

# 二. transformer 论文原理
首先推荐先阅读[周弈帆的博客解读transformer](https://zhouyifan.net/2022/11/12/20220925-Transformer/)， 达到能够理解以下要点

* 注意力机制: `q*K^T`做一次向量化查询，`sofmax(q*K^T / sqrt(d_model)) * V`完成查询结果的加权, sqrt(d_model)用于softmax缩放，将梯度集中在明显变化区域。每一次查询看一次key表，生成新的val特征，特征优化方向与loss下降方向一致。  
* 多头注意力设计: 折叠分组查询，使用更少的参数量，进行更多特征空间的交互。
* 注意力mask: 在seq_length维度保证当前查询只能看到自己以及之前的信息，模拟rnn的串行输出，在decoder中mask cross attention出现。
* 可训参数矩阵`Wq Wk Wv Wo` 实现类似自动化特征工程的效果，如对一个查询向量q计算`q * Wk^T`可以得到新的查询，查询优化方向和loss下降方向一致，torch中以nn.Linear
线性层表示这些矩阵。  
* FFN前馈神经网络, 隐藏层维度设置`4*d_model`，特征在更高维的隐空间交互，实现类似特征增强的效果, 4这个值目前看是约定俗成，没太多意义。
* pos embeding沿着seq_length和d_model对应的两个维度对token embedding加上pos数值，标记位置信息。简单理解一个特征矩阵Q中任意一个数值通过向前diff和向上diff可以锁定位置坐标，模型可以学到这种模式。
* token embedding矩阵为可学习矩阵，实现将一个token_id转换为对应embedding向量，维度为d_model。
* 训练阶段，对比rnn, transformer能够做到训练的并行，即输出一次性包含了所有input片段的next token，得力于attention mask的设计, 模拟信息串行。
* 预测阶段，与rnn相同，transformer自回归预测下一个token，当出现终止符则停止。
  
**论文链接[attendion all you need](https://arxiv.org/abs/1706.03762)**，论文模型结构为encoder-decoder的结构，两个组件的经典模型见第六节。

# 三. transformer 论文代码解读

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
根据[周弈帆的博客-PyTorch Transformer 英中翻译超详细教程](https://zhouyifan.net/2023/06/11/20221106-transformer-pytorch/)手撕一遍transformer的代码，了解各个组件设计以及代码设计风格。该代码基本与transformer论文结构相同，唯一的区别在于最后的`ouput head`是一个单独的线性层，与embeding层不共享权重。

# 四. llm模型训练流程及方法 - 完整训练一个GPT decoder-only的问答模型

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
推荐根据轻量化llm项目完整走一遍对话模型的开发[Minimind](https://github.com/jingyaogong/minimind)。
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
> 只要求跑通进行代码阅读的情况下，4Gb显存的卡将batch_size设置为1可以吃得消。

---

## 4.1 Pretrained
* prtrained的目的是让模型具备合理预测下一个token的能力，合理体现在能够根据一个字输出符合逻辑的话一段话，简而言之就是字接龙。

* prtrained的输入是`input_ids[:-1]`， 标签是`input_ids[1:]`，input_ids是指文字经过tokenize后的id list，如`我爱你 --> <s>我爱你<\s> --> [1, 2, 23, 4, 2]`，之所以输入与标签要错一位，目的在于实现预测下一个token的监督学习，例如输入文字是”我爱你啊“， 那么预测下一个token逻辑是`我 --预测--> 我爱; 我爱 --> 我爱你；我爱你 --> 我爱你啊`， 使用mask能对信息进遮掩，实现并行训练，即模型ouput中的每一个位置是由该位置之前的所有信息预测得到的, 初始的`ouput[0]`则由`<s>我`预测得到。

* prtrained的损失函数为corss_entropy，模型输出的logits维度为`(batch_size, max_seq_length, voc_size)`, `max_seq_length`为文字对截长补短的最大长度，`voc_size`为词表的token数量。损失计算的逻辑为对logits沿最后的轴进行softmax得到几率，形状不变；沿着max_seq_length取出label对应的token_id计算corss_entropy；由于所有label的真实长度不一定为max_seq_length，需要设置一个真实token_id的mask就行过滤。

---

## 4.2 SFT
* sft监督微调的目的是让模型具备对话能力，通过将prompt嵌入问答模版，如`用户<s>说:你是谁？</s>\n助手<s>回答:我是人工智能助手</s>\n`，构成一个新的语料微调pretrained模型，继续训练模型对这类模版的词语接龙能力。
  
* 对话模板通过引入特殊的字符，微调后能够让模型理解问题句柄，知道这是一个问题，从而触发预测问题后面的答案。

* sft与prtrained区别在于损失的计算以及训练的参数。sft只计算output中对应标签`回答: ***`的部分，其余部分不计入损失，但这些部分会在attention中被关注到；训练参数取决于不同的微调方法，常见> full-sft, lora, bitfit, preEmbed, prefix, adapter等。

### 01 full-sft 全量微调
全量微调是指使用pretrained初始化权重，使用较小的学习率对模型的全部参数进行训练，语料设计和损失设计同上。
  
### 02 lora-sft 低秩矩阵自适应微调
  [lora](https://arxiv.org/abs/2106.09685)对可学习矩阵**W（Wq Wk Wv Wo ...）**，增加两个低秩矩阵A和B，对输入进行矩阵乘法并相加`XW + XAB = X(W + AB) = XW‘`，`W'`为更新后的参数矩阵。假设原参数矩阵W的维度为`(d_k, d_model)`, 低秩矩阵A、B维度应该满足```(dk, r) (r, d_model)```，r为秩参数，r越大则A、B参数越多，W可更新的`△W`分布自由度更大。  
  
  相比全量微调lora需要的显存大大减小，但在小模型上训练速度不一定更快
  > 小模型forward过程耗时占比增加
  
### 03 其他微调方法
* PreEmbed，只微调token embedding参数矩阵，适应新的数据分布
  
* Prompt tuning，在输入token前增加特殊的提示token，只微调提示token的embeding向量参数，适合小模型适配下游任务

* P tunning，是Prompt tuning的进阶版，提示token可插入prompt的指定位置

* Prefix，在attention中`K=X * Wk，V=X * Wv`对X增加可学习前缀token embeding矩阵，作为虚拟的提示上下文, `K=[P; X]Wk V=[P; X]Wv`P是可学习的参数矩阵，维度(L, d_model)，L表示需要增加的提示前缀长度，是超参数。`[P; X]`表示在X输入矩阵开始位置拼接矩阵P。
  > prefix微调的是每一个transform层中的attention可学习前缀矩阵P，不同的层中，P不同。    
  
* Bitfit: 只微模型的偏置项，偏置项出现在所有线性层和Layernorma层中。    

* Adapter，在transform模块的多头注意力与输出层之后增加一个adpter层，只微调adpter参数。 adpter包含`下投影linear + nolinear + 上投影linear; skip-connect结构`， 中间结构类似lora变体为`nonlinear(XA)B`的结构，skip-connect结构保证的模型能力最多退化为原模型；由于改变了Laynorm输入的数据分布，Laynorm的scale参数也需要加入训练。  

---

## 4.3 preference opimized
  偏好对齐(优化)的目的是让模型的输出更加符合用户的习惯，包括文字逻辑、风格、伦理性、安全性等。  

### 01 rlhf
  pending 需要梳理强化学习的基础理论才能进阶

### 02 dpo
直接偏好优化(direct-preference-opimized)与rlhf不同，直接跳过了奖励模型的训练，根据偏好数据一步到位训练得到对齐模型。[论文](https://arxiv.org/abs/2305.18290)解读可以参考博客[人人都能看懂的DPO数学原理](https://mp.weixin.qq.com/s/aG-5xTwSzvHXN4B73mfKMA)  

筒体而言，dpo从rlhf总体优化目标出发```模型输出尽可能接近偏好标签，尽可能偏离非偏好标签，尽可能少偏离原模型输出```，推导最优奖励模型的显示解，代入奖励模型的损失函数，得到一个只与待训模型有关的损失函数，该函数就是偏好优化的目标。 

> 手撕dpo训练代码可以参考本仓库的`/minimind/5-dpo_train_self.py`

### 4.4 evalization
... pending 需要系统梳理多llm task才能进阶

---

# 五. RNN补充
在transformer出现后，在nlp的各任务中rnn逐渐被替代，但在一些结构化数据的时序预测仍广泛使用。
  
原理解读参照[吴恩达《深度学习专项》笔记（十四）：循环神经网络基础](https://zhouyifan.net/2022/09/21/DLS-note-14/)
  
代码实战参照[你的第一个PyTorch RNN模型——字母级语言模型](https://zhouyifan.net/2022/09/21/DLS-note-14-2/)  
  
* RNN基本原理可以概括为，通过维护一个中间状态`a(t)`，捕捉数据时序依赖关系。t时刻中间状态`a(t)`由t-1刻状态和t时刻的输入通过可学习参数矩阵W进行转换`a(t) = W([a(t-1), x(t)])`，`[a(t-1), x(t)]`表示横向拼接。t时刻输出由解码器对隐状态`a(t)`进行解码`y(t) = decoder(a(t))`。  
  
* RNN为了解决长时序依赖尾部数据难以获得首部数据信息问题， RNN变体模拟时序记忆的存储与衰减机制，代表性的有**GRU、LSTM**。
  > * GRU本质上考虑a(t)更新的偏好`a(t) = Wu * a(t) + (1 - Wu) a(t-1)`，增强中间隐状态对先前信息的留存空间。权重因子`Wu = sigmoid(W[a(t-1), x(t)]), W是可学习参数矩阵`
  
  > * LSTM需要维护两个中间隐状态，更新的机制也更加复杂，但整体思想与GRU相似。总之，GRU计算更高效，LSTM拟合能力更强。

* 基础RNN的结构只考虑单向的信息，t时刻的只能看到t时刻之前的信息编码，BRNN增加一个逆向传递结构（输入从后往前），实现t时刻双向信息编码。该抽象结构可以以基础RNN，GRU、LSTM为基模型进行搭建。不足之处在于对于需要完全输入信息后才能产生预测，不适用于实时输出的场景, 如实时翻译。
  
* 深层RNN可以叠加n个基rnn单元，自下往上，当前层的输出作为上一层的输入，需要维护n个隐状态。同时，可以增加更加复杂的输入编码和输出解码的结构，实现更复杂的特征工程和信息过滤，**适用于结构化时序数据的自动特征工程。**

---

# 六. 进阶1-经典transformer结构介绍

## GPT 
* introduce: decoder only结构，通过mask self-attention保证每个token只能看到上文信息，输出自回归预测下一个token。适用与输出为下一个关联token的所有sep2sep任务，如：问答，机器翻译，摘要生成，音乐生成等。

* prtrained: 采用自回归语言模型训练方式，见四.llm模型训练流程及方法。

* finetune: 采用sft监督指令微调，对每一条input-out数据对处理为特殊模版input进行自回归训练，见到四.llm模型训练流程及方法。

* practice: 见到`四minimind项目`。   
  
## Bert
* introduce: encoder only结构，self attendtion保证每个token可以看到上文和下信息，输出与句子整体语义相关，无法自回归预测next token。适用于输出为类别、数值的所有sep2sep，sep2val任务，如: 分类问题(情感分类，邮件分类, 多选问答，抽取问答...)，序列标注（词性标注 邮寄地址信息提取), 语义相似度...。对于bert的解读可以参考[链接](https://github.com/datawhalechina/learn-nlp-with-transformers)

* prtrained: 采用mask language和相邻句子判断进行预训练。  
  > * mask language随机遮掩token(15%, 其中10%被随机替换为其他token)，输出预测被遮掩的token，通过这种挖词填空促使模型也能理解上下文信息；
   
  > * 相邻句子判断，输入为句子+分隔标记+相邻句子，通过CLS位置的输出进行分类监督。这个训练步骤在后续的研究中逐渐淡化。  
  
  > * 特殊输入标记包括，类别标记`[CLS]`，句子分隔标记`[SEP]`，遮掩token标记`[MASK]`。`[CLS]`标记标记主要表征句子的整体语义，主要作为分类输出头的输入。  
  
  > * embedding由三类向量相加：`embeddings = words_embeddings + position_embeddings + token_type_embeddings`，token_type区分上句或下句，三者都是可学习参数，形状分别为`(voc_size, d_model), (max_len, d_model), (2, d_model)`。相加的含义可以用one-hot编码就行解释，等同于`onehot[word-hot, pos-hot, type-hot] * [W_word, W_pos, W_type]`。
  
  > * padding mask区分实际token和padding token，用于在softmax中归零padding token的权值，例如一个token查到paddingtoken，计算得到的注意力权重应该为0。

* finetune: 以bert作为backbone增加输出头，初始化pretained权重，只训输出网络或较以较小学习率全量微调即可达到不错的效果。

* practice: [bert中文分类](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)，快速理解整个bert模型结构，微调数据的加载方式和训练过程。
  > 在本仓库中增加地址文本的序列标注代码，见`/Bert-Chinese-Text-Classification-Pytorch/seqlabel/train.py` 
  
## T5 encoder-decoder 集大成者
* introduce: encoder-decoder结构，适用于所有的NLP任务包括序列标注、文本分类、机器翻译、摘要生成、问答。[论文地址](https://arxiv.org/abs/1910.10683)
  > Teacher Forcing的训练策略。本身用于rnn自回归任务中，训练时使用t时刻的真值作为t+1时刻的输入，但需要计算t时刻预测与真值的损失。
  > text2text框架适应
  > 相对位置编码
  > Teacher Forcing的训练策略

* pretreined：训练方法选择 mask and mask ratio，prefix的text2text方法
  > mask continous 策略: 对比mask一个token, mask连续token, token乱序恢复三类破坏方法，mask连续token效果最佳
  > mask 15%: 随机mask 15%的比例实验最佳效果
  

* finetune：

* prectice：



---

# 七. 进阶2-模型压缩
## 蒸馏

## 量化

## 剪枝

---

# 附录
## Model structure
### 1. BatchNorm vs LayerNorm vs RMSNorm

> 首先明确归一化的作用。数据经过每一层的变化和激活后，数据分布会不断向激活函数的上下限集中，此时激活函数所带来的梯度变化随着层变深而变小，最终出现梯度消失。

> 另一方面，机器学习建模的前提是训练与测试集独立同分布，当出现不同分布的数据时，模型可能降效。基于此，人为将数据拉倒相同分布有利于增强模型鲁棒性。同时将分布主体scale到0-1，集中在激活函数的明显变化区域，有利于解决深层网络梯度消失问题。

* batchnorm沿着特征维度对batch一视同仁进行归一化；layernorm沿着batch维度对特征一视同仁进行归一化；两者有两个可学习参数，rescale参数和偏置参数。

* rmsnorm是layernorm的改良版，去掉了去中心化的计算过程，提高了计算效率，只有一个可学习参数即rescale参数。

* batchnorm适用于卷积结构，训练时batchsize大均值与方差具备代表性。layernorm适transform、rnn结构，训练时batchsize小但是feature维度高；另一方面，图像数据是客观存在的表示，对每个sample的channel特征进行归一化具有实际意义。而自然语言的表示是人为构造的，通过embeding转换为数字表示，客观上并不存在，对每个sample特征维度进行归一化缺少实际意义。
  
### 2. sin-cos pos embedding vs ROPE vs 可学习的位置编码

* 可学习的位置编码好处是位置表征能力更强，但延展性差，无法处理出现超越编码长度的输入，bert模型使用该编码方式。

*  sin-cos pos对正余弦函数进行取值进行绝对位置编码，而rope则依次对相邻两个数值进行旋转变换，前后两者间具有相对的旋转位置关系，实现相对位置编码。两者都具有可延展性，rope在捕捉长序列的相对关系上更具有优势。
  
### 3. linear attention
  解决的问题，组件设计

---

## Large model fine tune

### Q-Lora
  应用场景 双重量化 + lora微调，牺牲计算的效率换内存

