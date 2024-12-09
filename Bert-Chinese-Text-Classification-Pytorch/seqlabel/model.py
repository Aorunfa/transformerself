# coding: UTF-8
import sys
sys.path.insert(0, '/home/chaofeng/Bert-Chinese-Text-Classification-Pytorch')
import torch
import torch.nn as nn
from pytorch_pretrained import BertForTokenClassification, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                            
        self.dev_path = dataset + '/data/dev.txt'                                  
                                    
        self.batch_size = 128                                           
        self.max_length = 32                                              
        self.num_classes = 18
        self.class_list = [str(i) for i in range(self.num_classes)]                                      
        
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'      
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 5                                             # epoch数
        self.learning_rate = 5e-5                                       # 学习率
        
        self.bert_path = '/home/chaofeng/Bert-Chinese-Text-Classification-Pytorch/bert_pretrain'    # pretrianed 
        
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(config.bert_path, 
                                                               num_labels=config.num_classes)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x, mask=None):                                                    
        logits = self.bert(x, attention_mask=mask)
        return logits

if __name__ == '__main__':
    cfg = Config('.')
    model = Model(cfg)
    model.load_state_dict(torch.load('/home/chaofeng/Bert-Chinese-Text-Classification-Pytorch/seqlabel/saved_dict/bert.ckpt', weights_only=True))
    model.eval()
    text = '广东省阳江市江城区王山一路'
    token = ['[CLS]'] + cfg.tokenizer.tokenize(text)
    token_ids = cfg.tokenizer.convert_tokens_to_ids(token)
    token_ids = torch.LongTensor([token_ids])
    with torch.no_grad():
        logits = model(token_ids)
        pred = torch.argmax(logits, -1)[..., 1:].view(-1).tolist()
    
    print(token_ids.shape)
    print(pred)
    print(dict(zip(token[1:], pred)))

