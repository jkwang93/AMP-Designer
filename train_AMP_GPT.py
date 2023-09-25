import os
import random
import time
import csv
import torch
import argparse
import numpy as np
import pandas as pd
# from rouge import Rouge
from tqdm.auto import tqdm
from torch.optim import AdamW, Adam
from transformers import get_scheduler, GPT2Config
from torch.utils.data import Dataset, DataLoader
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer
from torch.nn import CrossEntropyLoss

from early_stop.pytorchtools import EarlyStopping


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="", type=str, help='')
    parser.add_argument('--vocab_path', default="", type=str, help='')
    parser.add_argument('--save_model_path', default="small_save_model", type=str, help='')
    parser.add_argument('--final_model_path', default="small_final_model", type=str, help='')
    parser.add_argument('--train_raw_path', default='train_raw_data.txt', type=str, help='')
    parser.add_argument('--eval_raw_path', default='test_raw_data.txt', type=str, help='')
    parser.add_argument('--batch_size', default=128, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=1001, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=1000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=5e-3, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='print log steps')
    return parser.parse_args()


def calculate_loss_and_accuracy(outputs, labels, device):
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)
    not_ignore = shift_labels.ne(tokenizer.pad_token_id)
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    return loss, accuracy


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
    eval_data_list = []


    with open(train_data_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    print("数据总行数:{}".format(len(data)))

    random.shuffle(data)

    split_id = len(data)//10

    train_data = data[split_id:]
    eval_data = data[:split_id]



    for data_i in tqdm(train_data):
        data_i = data_i[0]
        data_list.append(tokenizer.encode(data_i))
        # data_list.append(tokenizer.encode(data_i, padding="max_length", truncation=True, max_length=34,
        #                                   return_special_tokens_mask=True, ))

    dataset = MyDataset(data_list)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)

    for data_i in tqdm(eval_data):
        data_i = data_i[0]
        eval_data_list.append(tokenizer.encode(data_i))
        # data_list.append(tokenizer.encode(data_i, padding="max_length", truncation=True, max_length=34,
        #                                   return_special_tokens_mask=True, ))

    eval_dataset = MyDataset(eval_data_list)
    eval_dataloader = DataLoader(dataset=eval_dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)

    return dataloader,eval_dataloader

def train(args, model, dataloader,eval_dataloader):
    num_training_steps = args.epochs * len(dataloader)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()
    batch_steps = 0
    early_stopping = EarlyStopping(patience=5, verbose=False)

    for epoch in range(args.epochs):
        epoch_loss_list = []
        print("\n")
        print("***********")
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        print("***********")
        print("\n")
        for batch in dataloader:
            batch_steps += 1
            inputs = {"input_ids": batch.to(device)}
            outputs = model(**inputs, labels=batch.to(device))
            # loss = outputs.loss
            loss, acc = calculate_loss_and_accuracy(outputs, batch.to(device), device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if batch_steps % args.log_step == 0:
                print("train epoch {}/{}, batch {}/{}, loss {}, accuracy {}".format(
                    epoch, args.epochs,
                    batch_steps,
                    num_training_steps,
                    loss, acc,
                ))

            epoch_loss_list.append(loss.cpu().detach().numpy())
        epoch_loss = evaluate(model,eval_dataloader)
        early_stopping(epoch_loss, model, args.save_model_path)

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.final_model_path)


def evaluate(model, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model = GPT2LMHeadModel.from_pretrained(args.save_model_path)

    model.to(device)
    model.eval()
    loss_list, acc_list = [], []
    batch_steps = 0
    # early_stopping = EarlyStopping(patience=5, verbose=False)

    with torch.no_grad():
        for batch in dataloader:
            batch_steps += 1
            inputs = {"input_ids": batch.to(device)}
            outputs = model(**inputs, labels=batch.to(device))
            loss, acc = calculate_loss_and_accuracy(outputs, batch.to(device), device)
            loss_list.append(float(loss))
            acc_list.append(float(acc))


            # print("eval batch {}/{}, loss {}, accuracy {}".format(
            #     batch_steps,
            #     len(dataloader),
            #     loss, acc))

    epoch_loss = np.mean(loss_list)
    # early_stopping(epoch_loss, model, args.save_model_path)


    print("loss: {},".format(np.mean(loss_list)),
          "accuracy: {}.".format(np.mean(acc_list)))
    return epoch_loss




def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    args = setup_args()
    args.model_path, args.vocab_path = '', './my_token/vocab.txt'
    args.train_raw_path = '../data/uniprot/space_canonical_uniport.csv'


    tokenizer = BertTokenizer(vocab_file=args.vocab_path)

    none = tokenizer.bos_token_id
    tokenizer.bos_token_id = tokenizer.cls_token_id
    tokenizer.eos_token_id = tokenizer.sep_token_id


    model_config = GPT2Config(
        architectures=["GPT2LMHeadModel"],  # pretrain的时候用来预加载模型
        model_type="GPT2LMHeadModel",  # 定义模型类型，导出给`AutoConfig`用，如果要上传到hub请必填
        # tokenizer_class="BertTokenizer",  # 定义tokenizer类型，导出给`AutoTokenizer`用，如果要上传到hub请必填
        vocab_size=25,
        n_positions=50,
        n_ctx=50,
        n_embd=768,
        n_layer=12,
        n_head=8,
        bos_token_id=tokenizer.bos_token_id,  # 前面构建的tokenizer的 PAD ID
        eos_token_id=tokenizer.eos_token_id,  # 前面构建的tokenizer的 PAD ID
        pad_token_id=tokenizer.pad_token_id,  # 前面构建的tokenizer的 PAD ID
        mask_token_id=tokenizer.mask_token_id,  # 前面构建的tokenizer的 PAD ID

        task_specific_params={
            "text-generation": {
                "do_sample": True,
                "max_length": 34
            }
        }
    )
    model = GPT2LMHeadModel(config=model_config)


    train_dataloader,eval_dataloader = data_loader(args, args.train_raw_path, tokenizer=tokenizer, shuffle=True)
    train(args, model, train_dataloader, eval_dataloader)
