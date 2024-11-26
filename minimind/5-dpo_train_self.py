import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import platform
import argparse
import time
import math
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import Transformer
from model.LMConfig import LMConfig
from model.dataset import DPODataset, dpo_collate_fn
import copy
from typing import Tuple
warnings.filterwarnings('ignore')
pwd = pwd = os.path.dirname(__file__)


"""
损失函数
数据输入流:
    q-a input --> 自回归每个数据的几率分布 --> 根据mask和label提取对应位置输出token_pos的logit, 得到logits-label
损失计算流
    得到logits-label, 根据label的token id提取logits每一个token分布位置的logit, 组合成与label等长的logits_new
    根据logits_new可以计算accept/reject的强度因子
    计算dpo loss, 同时输出accept与reject的偏离损失
"""

def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

def get_log_probs(logits, labels, mask):
    """
    logits shape(bsz, max_seq_len, tokennizer_max_length)
    labels shape(bsz, max_seq_len), labels can be accept or reject
    bsz can be 1
    """
    log_probs = F.log_softmax(logits, dim=-1)  
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    # get avg logitprob for each
    logitprobs = (log_probs_labels * mask).sum(-1) / mask.sum(-1)
    return logitprobs.unsqueeze(-1)

def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float = 0.1,
                    label_smoothing: float = 0.1,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def get_lr(it, all):
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch, wandb):
    start_time = time.time()
    model.train()
    ref_model.train()

    loss_item = 0
    for step, data in enumerate(train_loader):
        chosen_input_id = data['chosen_input_id'].to(args.device)
        rejected_input_id = data['rejected_input_id'].to(args.device)
        chosen_mask = data['chosen_mask'].to(args.device)
        rejected_mask = data['rejected_mask'].to(args.device)
        chosen_label = data['chosen_label'].to(args.device)
        rejected_label = data['rejected_label'].to(args.device)

        X_chosen = chosen_input_id[..., :-1].contiguous()
        X_rejected = rejected_input_id[..., :-1].contiguous()
        Y_chosen = chosen_input_id[..., 1:].contiguous()
        Y_rejected = rejected_input_id[..., 1:].contiguous()

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            # dpo model ward
            logits_chosen = model(X_chosen, Y_chosen).logits
            logits_rejected = model(X_rejected, Y_rejected).logits

            # ref model forward
            with torch.no_grad():
                logits_chosen_ref = ref_model(X_chosen, Y_chosen).logits.detach() 
                logits_rejected_ref = ref_model(X_rejected, Y_rejected).logits.detach() 
            
            chosen_mask = chosen_mask[..., 1:]
            rejected_mask = rejected_mask[..., 1:]
            chosen_label = chosen_label[..., 1:]
            rejected_label = rejected_label[..., 1:]

            # get logits_pros for chosen 
            policy_chosen_logps = get_log_probs(logits_chosen, chosen_label, chosen_mask)
            policy_rejected_logps = get_log_probs(logits_rejected, rejected_label, rejected_mask)
            reference_chosen_logps = get_log_probs(logits_chosen_ref, chosen_label, chosen_mask)
            reference_rejected_logps = get_log_probs(logits_rejected_ref, rejected_label, rejected_mask)


            loss, chosen_rewards, rejected_rewards = preference_loss(policy_chosen_logps, 
                                                                     policy_rejected_logps, 
                                                                     reference_chosen_logps, 
                                                                     reference_rejected_logps,
                                                                     beta=0.1)

        loss = loss.mean()
        scaler.scale(loss).backward()

        loss_item = (loss_item * (step) + loss.detach().item()) / (step + 1)
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    # loss.item(),
                    loss_item,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss_item,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/full_sft_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config, device): 
    # no lora fitune
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(pwd, 'model/minimind_tokenizer'))
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
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    model = model.to(device)
    return model, tokenizer

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    parser.add_argument("--out_dir", type=str, default="/home/chaofeng/minimind/dpo", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="Weights & Biases project name")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")

    args = parser.parse_args()

    lm_config = LMConfig()
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-LoRA-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    
    # for ddp
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None
    
    model, tokenizer = init_model(lm_config, args.device)
    # def init_model():
    #     device = args.device
    #     # Do model patching and add fast LoRA weights
    #     model_name_or_path = "/home/chaofeng/minimind/minimind-v1-small"
    #     tokenizer_name_or_path = "/home/chaofeng/minimind/minimind-v1-small"
    #     model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    #     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, use_fast=False)
    #     tokenizer.pad_token = tokenizer.eos_token
    #     model = model.to(device)
    #     return model, tokenizer
    # model, tokenizer = init_model()
    
    ref_model = copy.deepcopy(model)

    json_file = (os.path.join(pwd, 'dataset/dpo/dpo_train_data.json'))

    train_ds = DPODataset(json_file, tokenizer, prompt_max_len=max_seq_len // 2, answer_max_len=max_seq_len // 2)

    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), # filter(fn, iters)
        lr=args.learning_rate
    )

    if False and platform.system() != 'Windows' and float(torch.__version__.split('.')[0]) >= 2:
        Logger("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)
    

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
        ref_model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        ref_model = DistributedDataParallel(ref_model, device_ids=[ddp_local_rank])


    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
    

    model.eval()
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/full_sft_{lm_config.dim}{moe_path}.pth'

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save(state_dict, ckp)
    model.train()
    

