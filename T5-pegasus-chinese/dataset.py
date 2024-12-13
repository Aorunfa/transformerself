import torch
from transformers import T5Tokenizer
from torch.utils.data import Dataset, DataLoader
import json

pad_token_id = 0
text_length = 768   # 限定文本长度
title_length = 128  # 限定标题长度

class CNPData(Dataset):
    def __init__(self, file, tokenizer: T5Tokenizer):
        if isinstance(file, str):
            self.data = self.load_data(file)
        else:
            self.data = file
        self.data = [d for d in self.data if len(d['content']) < text_length - 2] 
        self.tokenizer = tokenizer

    def load_data(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)['data']
        return data

    def __len__(self, ):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        title = data['title']
        content = data['content']
        content = '摘要生成: \n' + content
        return content, title

def padding_length(subatch:list, max_length):
    lengths = [len(b) for b in subatch]
    new_subatch = []
    for l, b in zip(lengths, subatch):
        padding_len = max_length - l
        b = b + [pad_token_id] * padding_len
        new_subatch.append(b)
    return torch.LongTensor(new_subatch)



def get_collate_fn(tokenizer):
    def collect_fun(batch):
        # padding to max length
        contents = [b[0] for b in batch]
        labels = [b[1] for b in batch] # eos </s> will be auto passed 

        inputs = tokenizer(contents, max_length=text_length, truncation=True, return_tensors='pt', padding=True)
        labels = tokenizer(labels, max_length=title_length, truncation=True, return_tensors='pt', padding=True)

        return inputs, labels
    return collect_fun

if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained('./pretrained_model')
    file = './data/nlpcc_data.json'
    ds = CNPData(file=file, tokenizer=tokenizer)
    data = ds[0]
  
    loader = DataLoader(ds, num_workers=4, batch_size=64, collate_fn=get_collate_fn(tokenizer=tokenizer))
    # for batch in loader:
    #     print(batch['input']['input_ids'].shape)
    #     print(batch['input']['attention_mask'].shape)

    for inputs, labels in loader:
        print(inputs['input_ids'].shape)
        print(inputs['attention_mask'].shape)



        



