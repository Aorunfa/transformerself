import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


from dataset import get_dataloader
from model import Model, Config

# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from pytorch_pretrained.optimization import BertAdam
from torch.optim import lr_scheduler
from datetime import timedelta

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# 权重初始化
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def train(config, model, train_iter, dev_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}] # bias layer层不需要动量衰减
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=len(train_iter) * config.num_epochs)
    total_batch = 0                         
    dev_best_loss = float('inf')
    last_improve = 0                        
    flag = False                            
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels, mask) in enumerate(train_iter):
            trains = trains.to(cfg.device)
            labels = labels.to(cfg.device)
            mask = mask.to(cfg.device)

            outputs = model(trains, mask)
            model.zero_grad()
            # mask loss for real

            active_loss = mask.view(-1) == 1
            active_logits = outputs.view(-1, cfg.num_classes)[active_loss]
            active_labels = labels.view(-1)[active_loss]

            loss = F.cross_entropy(active_logits, active_labels)

            
            loss.backward()
            optimizer.step()
      
            # for i, param_group in enumerate(optimizer.param_groups):
            #        print(f"Learning rate for param group {i}: {param_group['lr']}")
            # scheduler.step()

            if total_batch % 100 == 0:
                # 过滤cls输出
                true = labels.data.cpu()
                predic = torch.max(outputs.data, -1)[1].cpu()
                mask = (mask.view(-1) == 1).data.cpu()
                true = true.view(-1)[mask]
                predic = predic.view(-1)[mask]
                train_acc = metrics.accuracy_score(true, predic)

                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, dev_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path, weights_only=True))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels, mask in data_iter:
            texts = texts.to(cfg.device)
            labels = labels.to(cfg.device)
            mask = mask.to(cfg.device)
    
            outputs = model(texts, mask)
            
            active_loss = mask.view(-1) == 1
            active_logits = outputs.view(-1, cfg.num_classes)[active_loss]
            active_labels = labels.view(-1)[active_loss]

            loss = F.cross_entropy(active_logits, active_labels)
            loss_total += loss

            labels = labels.data.cpu().view(-1).numpy()
            predic = torch.max(outputs.data, -1)[1].cpu().view(-1).numpy()
            mask = (mask.view(-1) == 1).data.cpu().numpy()
            # labels = labels.view(-1)[mask]
            # predic = predic.view(-1)[mask]
            labels = labels[mask]
            predic = predic[mask]
            
        
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


if __name__ == '__main__':
    cfg = Config(os.path.dirname(__file__))
    model = Model(cfg)
    model.to(cfg.device)
    train_loader = get_dataloader(cfg.train_path, cfg.tokenizer, cfg.max_length, num_workers=4, batch_size=cfg.batch_size)
    val_loader = get_dataloader(cfg.dev_path, cfg.tokenizer, cfg.max_length, num_workers=4, batch_size=cfg.batch_size)
    train(cfg, model, train_loader, val_loader)

