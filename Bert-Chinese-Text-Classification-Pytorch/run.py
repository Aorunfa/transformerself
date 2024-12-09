# coding: UTF-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='bert', help='choose a model: Bert, ERNIE')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = '/home/chaofeng/Bert-Chinese-Text-Classification-Pytorch/THUCNews'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    # print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)

"""
代码分析进度
数据处理 -- done 补短截长
数据加载 -- done 原生dataloader __iter__ + __next__ 方法组合使用
模型加载 -- done config覆写 bert.from_pretrined实例化 结构分析

--------------
tokenizer加载 -- doen, basictokenizer清洗中文字符 wordpicestokenizer实现token转换优先##模糊匹配
训练过程       --
验证过程

TODO:
权重初始化的类型以及效果，了解初始化的重要性在哪


bert 论文解读作为收尾
自主项目延展作为进阶：
    情感分类：舆情分析
    序列标注：邮寄地址解析
    多项选择：选择题

"""