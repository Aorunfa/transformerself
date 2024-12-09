from torch.utils.data import Dataset, DataLoader
import torch

CLS = '[CLS]'

class AddressData(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        super().__init__()
        self.texts, self.labels = self.read_data(file_path)
        self.tokenizer = tokenizer
        self.labels_map =  {'prov': 0,
                            'city': 1,
                            'district': 2,
                            'devzone': 3,
                            'town': 4,
                            'community': 5,
                            'village_group': 6,
                            'road': 7,
                            'roadno': 8,
                            'poi': 9,
                            'subpoi': 10,
                            'houseno': 11,
                            'cellno': 12,
                            'floorno': 13,
                            'assist': 14,
                            'distance': 15,
                            'intersection': 16,
                            'O': 17
                            }
        self.max_length = max_length
    
    def read_data(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        X = []
        Y = []
        one_x = ''
        one_y = ''
        for l in lines:
            if l == '\n':
                X.append(one_x[1:])
                Y.append(one_y[1:])
                one_x = ''
                one_y = ''
                continue
            l = l.strip().split(' ')
            x = l[0]
            y = l[1].split('-')[-1]
            one_x += ' ' + x
            one_y += ' ' + y
        if  one_y != '':
            X.append(one_x[1:])
            Y.append(one_y[1:])
        return X, Y
    
    def __len__(self, ):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        token = text.split(' ')
        label = label.split(' ')
        label = [0] + [self.labels_map[k] for k in label]
        # token 清洗
        token = [t.lower() for t in token] # lower
        token_clean = []
        for t in token:
            if self.tokenizer.vocab.get(t, None) is not None:
                token_clean.append(t)
            else:
                token_clean.append('[UNK]')
        token = [CLS] + token_clean

        # 截取长补充短
        token = token[:self.max_length]
        label = label[:self.max_length]
        token_ids = self.tokenizer.convert_tokens_to_ids(token)
        padding_len = self.max_length - len(token_ids)
        mask = [0] + [1] * len(token_ids[1:]) + [0] * padding_len
        token_ids = token_ids + [0] * padding_len
        label = label + [0] * padding_len

        token_ids = torch.LongTensor(token_ids)
        label = torch.LongTensor(label)
        mask = torch.LongTensor(mask)
        return token_ids, label, mask


def get_dataloader(file_path, tokenizer, max_length, num_workers, batch_size):
    ds = AddressData(file_path, tokenizer, max_length)
    return DataLoader(ds, num_workers=num_workers, batch_size=batch_size)

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/chaofeng/Bert-Chinese-Text-Classification-Pytorch')
    from pytorch_pretrained import BertModel, BertTokenizer
    bert_path = '/home/chaofeng/Bert-Chinese-Text-Classification-Pytorch/bert_pretrain'    # pretrianed         
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    file_path = '/home/chaofeng/Bert-Chinese-Text-Classification-Pytorch/seqlabel/data/train.txt'
    ds = AddressData(file_path, tokenizer, max_length=128)

    for token_ids, label, mask in ds:
        print(token_ids.shape)
        a = 1

    # token_ids, label, mask = ds[0]
    dsloader = get_dataloader(file_path, tokenizer, max_length=128, num_workers=8, batch_size=64)
    for data in dsloader:
        print(data[0].shape)
        





    

