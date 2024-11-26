import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length # 单个句子最大长度
        self.padding = 0

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        #
        sample = self.df.iloc[index]
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算padding损失
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)


class SFTDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, prompt_max_len=512, answer_max_len=256):
        super().__init__()
        self.df = df
        self.max_length = max_length
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        #
        self.tokenizer = tokenizer
        self.padding = 0
        self.bos_id = self.tokenizer('<s>assistant').data['input_ids']

    def __len__(self):
        return self.df.shape[0]

    def find_sublist_index(self, main_list, sub_list) -> int: # 找到最后一个sub_list的开始位置
        last_index = -1
        for i in range(len(main_list) - len(sub_list) + 1):
            if main_list[i:i + len(sub_list)] == sub_list:
                last_index = i
        return last_index

    def safe_eval(self, s):
        try:
            res = eval(s)
        except Exception as e:
            return []
        return res

    def __getitem__(self, index: int):
        #
        sample = self.df.iloc[index]
        history = self.safe_eval(sample['history'])
        q = str(sample['q'])
        a = str(sample['a'])

        messages = []
        for history_message in history:
            if len(history_message) <= 1:
                continue
            messages.append(
                {"role": 'user', "content": str(history_message[0])[:self.max_length // 2]}
            )
            messages.append(
                {"role": 'assistant', "content": str(history_message[1])[:self.max_length // 2]}
            )

        messages += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]

        # 加入问答特殊标志位， 注意力机制可以捕捉标志位含义 ************************
        new_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_id = self.tokenizer(new_prompt).data['input_ids'][:self.max_length]

        # 实际长度
        question_length = self.find_sublist_index(input_id, self.bos_id) + len(self.bos_id)
        
        # 最大长度的剩余部分
        padding_len = self.max_length - len(input_id)
        input_id = input_id + [self.padding] * padding_len
        mask_len = len(input_id) - question_length - padding_len # for answer length
        
        # 0表示不计算损失， 只计算输出答案的损失 ************************
        loss_mask = [0] * question_length + [1] * (mask_len) + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        loss_mask_tensor = torch.from_numpy(loss_mask)

        return X_tensor, Y_tensor, loss_mask_tensor


# class DPODataset(Dataset):
#     def __init__(self, json_file, tokenizer, prompt_max_len=512, answer_max_len=512):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.padding = 0
        
#         # self.bos_id = self.tokenizer('<s>assistant').data['input_ids']
#         self.bos_id = self.tokenizer('<s>assistant\n').data['input_ids']
#         self.eos_id = self.tokenizer('</s>\n').data['input_ids']
#         self.prompt_bos_id = self.tokenizer('<s>user\n').data['input_ids']

#         self.max_length = prompt_max_len + answer_max_len                           # 包含对话标志位        
#         self.prompt_max_len = prompt_max_len - len(self.prompt_bos_id) - len(self.eos_id)
#         self.answer_max_len = answer_max_len - len(self.bos_id) - len(self.eos_id)
#         self.label_max_len = answer_max_len
#         with open(json_file) as f:
#             self.data = json.load(f)

#     def __len__(self):
#         return len(self.data)

#     def get_chat_template(self, q, a):
#         q_prompt = self.prompt_bos_id + self.tokenizer(q).data['input_ids'][:self.prompt_max_len] + self.eos_id
#         answer = self.bos_id + self.tokenizer(a).data['input_ids'][:self.answer_max_len] + self.eos_id        
#         return q_prompt + answer
    


#     def find_sublist_index(self, main_list, sub_list) -> int:
#         last_index = -1
#         for i in range(len(main_list) - len(sub_list) + 1):
#             if main_list[i:i + len(sub_list)] == sub_list:
#                 last_index = i
#         return last_index
    
#     def padding_and_mask(self, input_id):
#         question_length = self.find_sublist_index(input_id, self.bos_id) + len(self.bos_id)
#         padding_len = self.max_length - len(input_id)
#         input_id = input_id + [self.padding] * padding_len        

#         mask_len = len(input_id) - question_length - padding_len
#         mask = [0] * question_length + [1] * (mask_len) + [0] * padding_len
#         return input_id, mask

#     def __getitem__(self, index: int):
#         sample = self.data[index]
#         prompt = str(sample['prompt'])
#         chosen = str(sample['chosen'])
#         rejected = str(sample['rejected'])
#         chosen_input_id = self.get_chat_template(prompt, chosen)
#         rejected_input_id = self.get_chat_template(prompt, rejected)
#         chosen_input_id, chosen_mask = self.padding_and_mask(chosen_input_id)
#         rejected_input_id, rejected_mask = self.padding_and_mask(rejected_input_id)
        
#         chosen_label = self.tokenizer(chosen).data['input_ids'][:self.answer_max_len] + self.eos_id
#         rejected_label = self.tokenizer(rejected).data['input_ids'][:self.answer_max_len] + self.eos_id        
#         chosen_label = chosen_label + [self.padding] * (self.label_max_len - len(chosen_label))
#         rejected_label = rejected_label + [self.padding] * (self.label_max_len - len(rejected_label))

#         data = {
#             'chosen_input_id': torch.tensor(chosen_input_id),
#             'rejected_input_id': torch.tensor(rejected_input_id),
#             'chosen_mask': torch.tensor(chosen_mask),
#             'rejected_mask': torch.tensor(rejected_mask),
#             'chosen_label': torch.tensor(chosen_label),
#             'rejected_label': torch.tensor(rejected_label)
#         }
#         return data
    
# def dpo_collate_fn(batch):
#     new_batch = {}
#     for k in batch[0].keys():
#         # print(k)
#         new_batch[k] = torch.concatenate([b[k][None, ] for b in batch])
#     return new_batch

class DPODataset(Dataset):
    def __init__(self, json_file, tokenizer, prompt_max_len=256, answer_max_len=256):
        super().__init__()
        self.tokenizer = tokenizer
        self.padding = 0
        self.bos_id = self.tokenizer('<s>assistant\n').data['input_ids']
        self.eos_id = self.tokenizer('</s>\n').data['input_ids']
        self.max_length = prompt_max_len + answer_max_len # 包含对话标志位        
        self.prompt_max_len = prompt_max_len 
        self.answer_max_len = answer_max_len
        with open(json_file) as f:
            self.data = json.load(f)
            np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def get_chat_template(self, q, a):
        q_prompt = self.tokenizer.apply_chat_template(
                                                [{"role": "user", "content": q}],
                                                tokenize=True,
                                                add_generation_prompt=True,
                                                        )
        if len(q_prompt) > self.prompt_max_len:
            q_prompt = q_prompt[:self.prompt_max_len - len(self.bos_id)] + self.bos_id
        
        answer = self.tokenizer(a).data['input_ids'] + self.eos_id   
        if len(answer) > self.answer_max_len:
            answer = answer[:self.answer_max_len - len(self.eos_id)] + self.eos_id

        # get padding answer label
        input_id = q_prompt + answer
        question_length = len(q_prompt)
        padding_length = self.max_length - len(input_id)
        mask_len = len(answer)
        input_id = input_id + [self.padding] *  padding_length
        mask = [0] * question_length + [1] * (mask_len) + [0] * padding_length
        label = [self.padding] * question_length + answer + [self.padding] *  padding_length

        return input_id, mask, label
    


    def __getitem__(self, index: int):
        sample = self.data[index]
        prompt = str(sample['prompt'])
        chosen = str(sample['chosen'])
        rejected = str(sample['rejected'])
        chosen_input_id, chosen_mask, chosen_label = self.get_chat_template(prompt, chosen)
        rejected_input_id, rejected_mask, rejected_label = self.get_chat_template(prompt, rejected)
        
        data = {
            'chosen_input_id': torch.tensor(chosen_input_id),
            'rejected_input_id': torch.tensor(rejected_input_id),
            'chosen_mask': torch.tensor(chosen_mask),
            'rejected_mask': torch.tensor(rejected_mask),
            'chosen_label': torch.tensor(chosen_label),
            'rejected_label': torch.tensor(rejected_label)
        }
        return data

    # def __getitem__(self, index: int):
    #     sample = self.data[index]
    #     prompt = str(sample['prompt'])
    #     chosen = str(sample['chosen'])
    #     rejected = str(sample['rejected'])
    #     chosen_input_id = self.get_chat_template(prompt, chosen)
    #     rejected_input_id = self.get_chat_template(prompt, rejected)
    #     chosen_input_id, chosen_mask = self.padding_and_mask(chosen_input_id)
    #     rejected_input_id, rejected_mask = self.padding_and_mask(rejected_input_id)
        
    #     chosen_label = self.tokenizer(chosen).data['input_ids'] + self.eos_id
    #     rejected_label = self.tokenizer(rejected).data['input_ids'] + self.eos_id
    #     if len(chosen_label) > self.answer_max_len:
    #         chosen_label = chosen_label[:self.answer_max_len - len(self.eos_id)] + self.eos_id
    #     if len(rejected_label) > self.answer_max_len:
    #         rejected_label = rejected_label[:self.answer_max_len - len(self.eos_id)] + self.eos_id
      
    #     if sum(chosen_mask) != len(chosen_label) + 1:
    #         print('aaa')
        
    #     chosen_label = chosen_label + [self.padding] * (self.answer_max_len - len(chosen_label))
    #     rejected_label = rejected_label + [self.padding] * (self.answer_max_len - len(rejected_label))

        
    #     data = {
    #         'chosen_input_id': torch.tensor(chosen_input_id),
    #         'rejected_input_id': torch.tensor(rejected_input_id),
    #         'chosen_mask': torch.tensor(chosen_mask),
    #         'rejected_mask': torch.tensor(rejected_mask),
    #         'chosen_label': torch.tensor(chosen_label),
    #         'rejected_label': torch.tensor(rejected_label)
    #     }
    #     return data
    
def dpo_collate_fn(batch):
    new_batch = {}
    for k in batch[0].keys():
        # print(k)
        new_batch[k] = torch.concatenate([b[k][None, ] for b in batch])
    return new_batch

if __name__ == "__main__":
    # # pretrain dataset
    # from transformers import AutoTokenizer
    # data_path = '/home/chaofeng/minimind/dataset/pretrain_data.csv'
    # df = pd.read_csv(data_path, nrows=10)
    # df = df.sample(frac=1.0)
    # tk_path = '/home/chaofeng/minimind/model/minimind_tokenizer'
    # tokenizer = AutoTokenizer.from_pretrained(tk_path)
    # train_ds = PretrainDataset(df, tokenizer, max_length=512)
    # data = train_ds[1]
    # """
    # dataset构建过程:
    # 词语表加载
    # 语料读取
    # 每一条语料转换为idx(tokenizer)
    # 输出X, Y, mask for padding
    # """

    # SFT dataset
    # from transformers import AutoTokenizer
    # # data_path = '/home/chaofeng/minimind/dataset/sft_data_single.csv'
    # data_path = '/home/chaofeng/minimind/dataset/sft_data_multi.csv'
    # df = pd.read_csv(data_path, nrows=10)
    # #df = df.sample(frac=1.0)
    # tk_path = '/home/chaofeng/minimind/model/minimind_tokenizer'
    # tokenizer = AutoTokenizer.from_pretrained(tk_path)
    # train_ds = SFTDataset(df, tokenizer, max_length=512)
    # data = train_ds[1]

    # DPO dataset
    json_file = '/home/chaofeng/minimind/dataset/dpo/dpo_train_data.json'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/chaofeng/minimind/model/minimind_tokenizer')

    train_ds = DPODataset(json_file, tokenizer)
    # data = train_ds[1]
    # length = []
    # for i in range(2000):
    #     length.append(train_ds[i])
    
    #print('max ', max(length))
    #print('min ', min(length))

    from torch.utils.data import DataLoader, DistributedSampler
    train_loader = DataLoader(train_ds,
               batch_size=8,
               num_workers=8,
               collate_fn=dpo_collate_fn)
    for batch in train_loader:
        print(batch['chosen_input_id'].shape)



