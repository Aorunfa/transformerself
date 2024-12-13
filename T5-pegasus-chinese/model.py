from transformers import T5ForConditionalGeneration, T5Tokenizer, MT5ForConditionalGeneration
import  torch.nn as nn

class MengziBase(nn.Module):
    def __init__(self, pretrained_path, model_type='mt5'):
        super().__init__()
        if model_type == 'mt5':
            self.model = MT5ForConditionalGeneration.from_pretrained(pretrained_path)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(pretrained_path)

    def forward(self, inputs, labels=None, beam_search=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        if labels is not None:
            train_labels = labels['input_ids'].contiguous().to(device)
            train_labels_mask = labels['attention_mask'].to(device)   

            decoder_input_ids = train_labels.new_zeros(train_labels.shape)
            decoder_input_ids[..., 1:] = train_labels[..., :-1].clone()    # first place for padding

            decoder_attention_mask = train_labels_mask.new_zeros(train_labels_mask.shape)
            decoder_attention_mask[..., 1:] = train_labels_mask[..., :-1].clone()
            decoder_attention_mask[..., 0] = 1  # first padd is for predict the label first 
            outputs = self.model(input_ids=input_ids
                                 , attention_mask=attention_mask
                                 , decoder_input_ids=decoder_input_ids
                                 , decoder_attention_mask=decoder_attention_mask
                                 , labels=train_labels)
            
            ### 官方推荐方式
            # train_labels = labels['input_ids'].contiguous().to(device)
            # train_labels[train_labels.eq(0)] = -100 # -100 for auto mask
            # outputs = self.model(input_ids=input_ids,
            #                      attention_mask=attention_mask,
            #                      labels=train_labels)
            return outputs.loss

        else:
            if beam_search:
                summary_ids = self.model.generate(input_ids,
                                                num_beams=4,              # 束搜索法
                                                no_repeat_ngram_size=2,   # 确保不重复
                                                min_length=10,            # 长度限制
                                                max_length=64,
                                                early_stopping=True,
                                                eos_token_id=1
                                                )
            else:
                ## greedy decode default 
                summary_ids = self.model.generate(input_ids,
                                                min_length=10,    
                                                max_length=64,
                                                eos_token_id=1,
                                                )                # 默认use_cache
            return summary_ids