from torch.utils.data import DataLoader, Dataset
import re
import torch

PADDING = 0
ENCODING_MAP = ['<s>', '</s>'] + [chr(ord('a') + i) for i in range(26)]
LETTER_MAP = dict(zip(ENCODING_MAP, [i for i in range(len(ENCODING_MAP))] 
                      ))
def tokenize(s, use_onhot=True):
    letters = list(s)
    letters = ['<s>'] + letters + ['</s>']
    letter_ids = [LETTER_MAP[l] for l in letters]
    if use_onhot:
        x = torch.zeros(len(letter_ids), len(LETTER_MAP))
        for i, ids in enumerate(letter_ids):
            x[i, ids] = 1
        x = x.reshape(1, len(letter_ids), -1)
    else:
        x = torch.tensor(letter_ids).reshape(1, -1)
    return x, torch.tensor(letter_ids).reshape(1, -1, 1)


class WordDataset(Dataset):
    def __init__(self, input_path, max_len, use_onhot):
        super().__init__()
        with open(input_path, 'r') as f:
            data = f.readlines()
        self.data = [re.sub(r'[^a-zA-Z]', '', l) for l in data]
        self.data = [l for l in self.data if len(l) > 0]
        self.max_len = max_len
        self.use_onhot = use_onhot
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        letters = ['<s>'] + list(self.data[index].lower())
        letters = letters[: self.max_len - 1] + ['</s>']
        letter_ids = [LETTER_MAP[l] for l in letters]
        padding_len = self.max_len - len(letter_ids)
        mask = [1] * len(letter_ids) + [0] * padding_len
        letter_ids = letter_ids + [PADDING] * padding_len
        if self.use_onhot:
            x = torch.zeros(self.max_len, len(LETTER_MAP))
            for i, ids in enumerate(letter_ids):
                x[i, ids] = 1
        else:
            x = torch.tensor(letter_ids, dtype=torch.long)
        
        y = torch.tensor(letter_ids, dtype=torch.long)
        return  {'words': x, 'labels': y, 'mask': torch.tensor(mask, dtype=torch.long)}

def collect_fun(batch):
    x = torch.concat([b['words'][None, ] for b in batch])
    y = torch.concat([b['labels'][None, ] for b in batch])
    mask = torch.concat([b['mask'][None, ] for b in batch])
    return x, y, mask


if __name__ == '__main__':
    # print(ENCODING_MAP)
    # print(LETTER_MAP)
    wd = WordDataset('/home/chaofeng/DL-Demos/dldemos/BasicRNN/data/aclImdb/imdb.vocab',
                max_len=10,
                use_onhot=True)
    # x,m = wd[0]
    # print(x.shape)
    # print(m.shape)
    # print(x)
    # print(m)

    dl = DataLoader(wd, batch_size=64, num_workers=8, collate_fn=collect_fun)
    for x, y, mask in dl:
        print(x.shape)
        print(y.shape)
        print(mask.shape)
        
        

        
        
        



