import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from transformers import T5Tokenizer
import torch
import json
from dataset import CNPData, get_collate_fn
from torch.utils.data import DataLoader, random_split
from torch.optim.adam import Adam
from torch.optim import lr_scheduler
from model import MengziBase
from rouge import Rouge
import tqdm

class train_config(object):
    def __init__(self):
        self.epochs = 3
        self.lr = 2e-5
        self.lr_final = 1e-6
        self.warmup = 100

        self.device = 'cuda'
        self.accumulate = 4
        self.grad_clip = 1.


        self.test_step = 1000
        self.save_step = self.test_step


        self.save_path = './mt5_menzi_sft_%d.pth'


def print_loss(step_num, loss):
    l = loss.data.item()
    print("step: {:0>8d}{:>8s} loss: {:.4f}".format(step_num, '', l))
    


def train(model: MengziBase, config: train_config, train_loader, test_loader):
    model.to(config.device)    
    # print(test(model, test_loader))
    model.train()

    # # torch optimer
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    
    # optimizer = Adam(optimizer_grouped_parameters,
    #                  lr=config.lr
    #                  )
    
    
    # 使用transformer Adafactor进行训练,可以减少一半显存消耗
    from transformers.optimization import Adafactor, AdafactorSchedule
    optimizer = Adafactor(model.parameters(),  lr=config.lr, relative_step=False, scale_parameter=True, warmup_init=False,)



    scheduler_warmup = lr_scheduler.LinearLR(optimizer, 
                                             start_factor=0.1, 
                                             end_factor=1, 
                                             total_iters=config.warmup)

    scheduler = lr_scheduler.LinearLR(optimizer, 
                                      start_factor=1, 
                                      end_factor=0.01, 
                                      total_iters=config.epochs * len(train_loader) - config.warmup)
    



    for epoch in range(config.epochs):
        for step_num, (inputs, labels) in enumerate(train_loader):
            step_num += epoch * len(train_loader) + 1
            if step_num <= config.warmup:
                scheduler_warmup.step()

            # for i, param_group in enumerate(optimizer.param_groups):
            #     print(f"Learning rate for param group {i}: {param_group['lr']}") 
            
            else:
                scheduler.step()
            
            # optimizer.zero_grad()
            
            loss = model(inputs, labels)    # cross entropy
            
            optimizer.zero_grad()
            loss.backward()

            if step_num % config.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
                optimizer.step()
                print_loss(step_num, loss)
            
            if step_num % config.test_step == 0:
                # for validation
                metric = test(model, test_loader)
                model.train()
                print(metric)
            
            if step_num % config.save_step == 0:
                model.eval()
                torch.save(model.state_dict(), config.save_path % step_num)
                model.train()

    model.eval()
    torch.save(model.state_dict(), config.save_path % step_num)
    metric = test(model, test_loader)
    print(metric)

   
@torch.no_grad()
def test(model, test_loader):
    global tokenizer
    rouge = Rouge()
    model.eval()
    scores_tol = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }


    for inputs, labels in tqdm.tqdm(test_loader):
        decode_ids = model(inputs)
        label_ids = labels['input_ids']
        
        decode_preds = tokenizer.batch_decode(decode_ids, skip_special_tokens=True)
        decode_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        decode_preds = [" ".join(p) for p in decode_preds]   # 坑爹，必须加上空格，匹配rouge的格式
        decode_labels = [" ".join(l) for l in decode_labels]
        
        for i, p in enumerate(decode_preds):
            if p == '':
                print(i)
                print(decode_ids[i])
                print(decode_labels[i])
                break
        try:
            scores = rouge.get_scores(decode_preds, decode_labels, avg=True, ignore_empty=True)
            scores = {
                        "rouge-1": scores["rouge-1"]["f"],
                        "rouge-2": scores["rouge-2"]["f"],
                        "rouge-l": scores["rouge-l"]["f"],
                    }
            
        except Exception as e: # for empty preds
            print(e)
            scores = {
                        "rouge-1": 0,
                        "rouge-2": 0,
                        "rouge-l": 0,
                    }
        
        for k, v in scores.items():
            scores_tol[k] = v + scores_tol[k]


    return {k: v / len(test_loader) for k, v in scores_tol.items()}



if __name__ == '__main__':
    # preained_path = './pretrained_model'
    # model = MengziBase(preained_path, model_type='t5')

    preained_path = './pretrained_model'
    model = MengziBase(preained_path)

    tokenizer = T5Tokenizer.from_pretrained(preained_path)
    data_file = './data/nlpcc_data.json'
    
    ds = CNPData(data_file, tokenizer=tokenizer)
    train_ds, test_ds = random_split(ds, lengths=[0.8, 0.2], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, num_workers=4, batch_size=8, collate_fn=get_collate_fn(tokenizer))
    test_loader = DataLoader(test_ds, num_workers=4, batch_size=128, collate_fn=get_collate_fn(tokenizer))

    config = train_config()
    train(model, config, train_loader, test_loader)

