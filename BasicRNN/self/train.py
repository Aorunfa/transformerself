import torch.amp
from dataset import WordDataset, collect_fun, LETTER_MAP
from model import RNN1, RNN2, RNN3
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import time

# rnn = RNN1(hidden_units=32, embdeding_len=len(LETTER_MAP))
# wd = WordDataset('/home/chaofeng/DL-Demos/dldemos/BasicRNN/data/aclImdb/imdb.vocab',
#                 max_len=16,
#                 use_onhot=True)


# rnn = RNN2(max_len=len(LETTER_MAP), hidden_units=32, embdeding_len=64)
rnn = RNN3(max_len=len(LETTER_MAP), hidden_units=32, embdeding_len=64)
wd = WordDataset('/home/chaofeng/DL-Demos/dldemos/BasicRNN/data/aclImdb/imdb.vocab',
                max_len=16,
                use_onhot=False)


train_dataloader = DataLoader(wd, batch_size=64, num_workers=4, collate_fn=collect_fun)
creterion = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01)

epochs = 10
epoch_batch_num = len(train_dataloader)
accumulate = 10
step = 0

# 学习率
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=epochs * epoch_batch_num // accumulate)

device = 'cuda:3'
rnn.to(device)

### add mix prcision ###
amp = torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True) 
scaler = torch.amp.GradScaler(device=device)  # 防止amp梯度下溢超边界
######################
for epoch in range(epochs):
    for x, y, mask in train_dataloader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        with amp:
            output = rnn(x)
            B, L, _ = output.shape
            loss = creterion(output.reshape(B * L, -1), y.reshape(B * L))
            loss = (loss * mask.reshape(B * L)).mean()
            step += 1
        scaler.scale(loss).backward()   # scale grade
        
        if step % accumulate == 0:
            scaler.unscale_(optimizer)  # resuum scaled grade
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1) # 梯度爆炸
            
            scaler.step(optimizer)
            scaler.update()             # refresh
            # optimizer.step()

            print(loss.item())
            lr_scheduler.step()
            optimizer.zero_grad()

# save optim and model 
check_point = {
    'model': rnn.state_dict(),
    'optimizer': optimizer.state_dict(),
    'train_epoch': epochs,
    'timestamp': time.time(),
    }
torch.save(check_point, '/home/chaofeng/DL-Demos/dldemos/BasicRNN/self/checkpoint/last_lstm.pt')
