import os
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
import torch
from model.model import Transformer
from model.LMConfig import LMConfig

warnings.filterwarnings('ignore')


def init_model():
    device = 'cuda:0'
    # Do model patching and add fast LoRA weights
    model_name_or_path = "/home/chaofeng/minimind/minimind-v1-small"
    tokenizer_name_or_path = "/home/chaofeng/minimind/minimind-v1-small"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    return model, tokenizer

def init_model2(lm_config, device): 
    # no lora fitune
    tokenizer = AutoTokenizer.from_pretrained(os.path.join('/home/chaofeng/minimind', 'model/minimind_tokenizer'))
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = Transformer(lm_config)
    ckp = '/home/chaofeng/minimind/sft_full/full_sft_512.pth'
    state_dict = torch.load(ckp, map_location=device)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    print(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    model = model.to(device)
    return model, tokenizer

if __name__ == '__main__':
    lm_config = LMConfig()
    device = 'cuda:0'
    model, tokenizer = init_model2(lm_config, device)
    tokenizer.pad_token = '<unk>'
    # model, tokenizer = init_model()
    training_config = DPOConfig(
        output_dir="/home/chaofeng/minimind/dpo_raw",
        per_device_train_batch_size=16,
        remove_unused_columns=False,
        report_to="none",
        save_steps=20000,
        learning_rate=4e-5
    )

    dataset_path = '/home/chaofeng/minimind/dataset/dpo/dpo_train_data.json'
    train_dataset = load_dataset('json', data_files=dataset_path)

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_config,
        beta=0.1,
        train_dataset=train_dataset['train'],
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=512
    )
    dpo_trainer.train()
