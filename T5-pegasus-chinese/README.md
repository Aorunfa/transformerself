# t5-pegasus-chinese
中文文本摘要生成微调实战

# 环境安装
```bash
cd T5-pegasus-chinese
pip install -r requirements.txt
```

# 下载预训练模型
下载中文t5或mt模型放在pretrained_model目录下，任意下载一个即可。目录下为模型权重，模型配置和词表等相关配置文件，例如
   - pytorch_model.bin
   - config.json
   - tokenizer.json  
   
   > 中文孟子t5预训练模型下载链接，分享来自renmada
   (百度网盘 提取码：fd7k)[https://pan.baidu.com/s/1JIjEEyX-dgmqpQdL7aNbAw]，
   (Google Drive)[https://drive.google.com/file/d/18Y5LVghAGbz7ys0noii1eM1yDtFmW490/view?usp=sharing]
   
   > 中文孟子mt5摘要微调模型下载链接
   (huggingface)[https://huggingface.co/yihsuan/mt5_chinese_small]

# 下载数据
下载中文摘要-文本对数据集，解压在data目录下，目录结构如
   - nlpcc_2017.json
   > 下载链接(huggingface)[https://huggingface.co/datasets/supremezxc/nlpcc_2017]


# 微调
单卡微调，需求至少6GB显存
```python
python ./train.py
```

# 推理 
```python
python ./infer.py
```