

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN1(nn.Module):
    def __init__(self, hidden_units, embdeding_len):
        super().__init__()
        self.hidden_units = hidden_units
        self.embdeding_len = embdeding_len
        self.Wa = nn.Linear(hidden_units + embdeding_len, hidden_units)
        self.Wy = nn.Linear(hidden_units, embdeding_len)
        self.g = nn.Tanh()
    
    def forward(self, words: torch.Tensor):
        # words one-hot , shape(batch, max_len, embdeding_len)
        bsz, _, _ = words.shape
        words = words.transpose(0, 1) # shape(max_len, batch, embdeding_len)
        output = torch.zeros_like(words).to(words.device)
        
        # init first input
        a = torch.zeros(bsz, self.hidden_units).to(words.device)

        # recursive
        for i, word in enumerate(words):
            # word shape(batch, embdeding_len)
            a = self.Wa(torch.concat([a, word], dim=1))
            output[i] = self.Wy(a)
        output = output.transpose(0, 1) # reshape(batch, max_len, embdeding_len)
        return output

class RNN2(nn.Module):
    """
    use GRU and embedding
    """
    def __init__(self, max_len, hidden_units, embdeding_len, dropout=0.1):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout)
        self.embeding = nn.Embedding(max_len, embdeding_len)
        self.rnn = nn.GRU(embdeding_len, hidden_units, num_layers=1, batch_first=True)
        self.decoder = nn.Linear(hidden_units, embdeding_len)
        


    def forward(self, words: torch.Tensor):
        # words one-hot , shape(batch, max_len), include word index
        bsz, _= words.shape
        words = self.embeding(words)            # shape(batch, max_len, embdeding_len)s
        words = self.dropout(words)
        
        # init first input, default 0
        a = torch.zeros(1, bsz, self.hidden_units, device=words.device)
        
        # self.gru forward for all word once time
        output, a = self.rnn(words, a)  # output is the raw a(t), a is the final weight a(t)
                                        # output shape(batch, max_len, hidden_units)
                                        # a shape(1, batch, hidden_units)
        output = self.decoder(output)
        return output

class RNN3(nn.Module):
    """
    use LSTM and embedding
    """
    def __init__(self, max_len, hidden_units, embdeding_len, dropout=0.1):
        super().__init__()        
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout)
        self.embeding = nn.Embedding(max_len, embdeding_len)
        self.lstm = nn.LSTM(embdeding_len, hidden_units, num_layers=1, batch_first=True) # args same as gru
        self.decoder = nn.Linear(hidden_units, embdeding_len)
        
        

    def forward(self, words: torch.Tensor):
        # words one-hot , shape(batch, max_len), include word index
        bsz, _= words.shape
        words = self.embeding(words)            # shape(batch, max_len, embdeding_len)s
        words = self.dropout(words)
        
        # init first input, default 0
        # a = torch.zeros(1, bsz, self.hidden_units, device=words.device)
        
        # self.gru forward for all word once time
        output, hidden = self.lstm(words)  # output is the raw a(t), a is the final weight a(t)
                                        # output shape(batch, max_len, hidden_units)
                                        # a shape(1, batch, hidden_units)
        output = self.decoder(output)
        return output


if __name__ == '__main__':
    ### RNN1
    # words = torch.ones(8, 50, 27)
    # rnn = RNN1(32, 27)
    # output = rnn(words)
    # print(output.shape)
    # print(output)

    # ### RNN2
    # words = torch.ones(8, 50, dtype=torch.long)
    # rnn = RNN2(27, 32, 64)
    # output = rnn(words)

    # print(output.shape)
    # # print(output)


    #### test
    ## RNN1
    from dataset import LETTER_MAP, tokenize
    rnn = RNN1(hidden_units=32, embdeding_len=len(LETTER_MAP))
    ckpt = torch.load('/home/chaofeng/DL-Demos/dldemos/BasicRNN/self/checkpoint/last.pt', weights_only=True)
    rnn.load_state_dict(ckpt['model'])
    rnn.eval()
    with torch.no_grad():
        input = 'apple'
        input = 'appll'
        input = 'raaaly'
        input, label = tokenize(input)
        output = rnn(input)
        # get logits
        output = output.softmax(-1).gather(-1, label)
        probs = output.flatten()[1:-1]
        p = torch.ones(1)
        for prob in probs:
            p *= prob
        print(p)



    
        
        
        


            
        


