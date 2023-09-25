import os
import random
import time
import torch
import argparse
import numpy as np
import pandas as pd
# from rouge import Rouge
# from torchinfo import summary
from tqdm.auto import tqdm
from torch.optim import AdamW, Adam
from transformers import get_scheduler, GPT2Config
from torch.utils.data import Dataset, DataLoader
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

# from early_stop.pytorchtools import EarlyStopping
from soft_prompt_embedding import SoftEmbedding


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)

def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).to(device)
    return torch.autograd.Variable(tensor)

def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))

def calculate_likelihood_loss(outputs, labels, device, tokenizer,n_tokens):
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., n_tokens:-1, :].contiguous()
    shift_labels = labels[..., n_tokens+1:].contiguous().to(device)

    #todo replace the tokens after end tokens with pad_token_id

    # Flatten the tokens
    # loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
    # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    log_prob = F.log_softmax(shift_logits)

    loss = NLLLoss(log_prob.view(-1, log_prob.size(-1)), shift_labels.view(-1))

    loss=loss.view(shift_labels.size())
    loss = torch.sum(loss,dim=1)


    return loss

def NLLLoss(inputs, targets):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).to('cuda')
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss

def collate_fn(batch):
    input_ids = []
    input_lens_list = [len(w) for w in batch]
    max_input_len = max(input_lens_list)
    for btc_idx in range(len(batch)):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([tokenizer.pad_token_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)

def data_loader(args, train_data_path, tokenizer, shuffle):
    data_list = []

    train_data = pd.read_csv(train_data_path, header=None).values.flatten().tolist()
    print("数据总行数:{}".format(len(train_data)))

    for data_i in tqdm(train_data):
        data_list.append(tokenizer.encode(data_i, padding="max_length", truncation=True, max_length=34,
                                          return_special_tokens_mask=True, ))

    dataset = MyDataset(data_list)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)
    return dataloader

def top_k_top_p_filtering(logits: torch.FloatTensor, top_k: int = 0, top_p: float = 1.0,
                          filter_value: float = -float("Inf"),
                          min_tokens_to_keep: int = 1,
                          ) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_p = float(top_p)
    if top_k > 0:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if 0 < top_p <= 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


def decode(matrix,tokenizer):
    all_seqs = []
    seqs_len = []
    for i in matrix:
        chars = []
        i = tokenizer.convert_ids_to_tokens(i)
        for j in i:
            if j == '[SEP]':
                break
            chars.append(j.upper())
        seq = "".join(chars)
        all_seqs.append(seq)
        seqs_len.append(len(seq))
    return all_seqs, seqs_len




def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def prompt_model_loader(bin_path,model_dir):
    prompt_model_load = torch.load(bin_path)

    model = GPT2LMHeadModel.from_pretrained(model_dir)
    s_wte = SoftEmbedding(model.get_input_embeddings(),
                          n_tokens=10,
                          initialize_from_vocab=True)

    s_wte.learned_embedding.data = prompt_model_load['transformer.wte.learned_embedding']
    s_wte.wte.weight.data = prompt_model_load['transformer.wte.wte.weight']

    del prompt_model_load
    model.set_input_embeddings(s_wte)

    return model