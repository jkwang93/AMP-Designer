import csv
import os
import random
import time
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.optim import AdamW, Adam
from transformers import get_scheduler, GPT2Config
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, DistributedSampler
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer
from torch.nn import CrossEntropyLoss

from pytorchtools import EarlyStopping
from soft_prompt_embedding import SoftEmbedding


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)


def collate_fn(batch):
    input_ids = []
    input_lens_list = [len(w) for w in batch]
    max_input_len = max(input_lens_list)
    for btc_idx in range(len(batch)):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([tokenizer.pad_token_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)


def load_and_cache_examples(args, filepath, tokenizer):
    data_pos = []
    data_neg = []
    data = []
    data_taski = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        if args.sup_data_num <= 0:
            if not args.balanced:
                for row in reader:
                    data.append([row['comment_text'], row['id'], row['label']])
            else:
                for row in reader:
                    if int(row['label']) == 1:
                        data_pos.append([row['comment_text'], row['id'], row['label']])
                    else:
                        assert (int(row['label']) == 0)
                        data_neg.append([row['comment_text'], row['id'], row['label']])

                if len(data_pos) > len(data_neg):
                    data_neg_expand = data_neg * (len(data_pos) // len(data_neg))
                    data = data_pos + data_neg_expand + random.sample(data_neg, len(data_pos) - len(data_neg_expand))
                elif len(data_neg) > len(data_pos):
                    data_pos_expand = data_pos * (len(data_neg) // len(data_pos))
                    data = data_neg + data_pos_expand + random.sample(data_pos, len(data_neg) - len(data_pos_expand))
                else:
                    data = data_neg + data_pos
        else:
            for row in reader:
                if not row['label'] in data_taski.keys():
                    data_taski[row['label']] = []
                data_taski[row['label']].append([row['comment_text'], row['id'], int(row['label']), int(row['label'])])
            for label in data_taski.keys():
                if len(data_taski[label]) > args.sup_data_num:
                    add_data = random.sample(data_taski[label], args.sup_data_num)
                else:
                    add_data = data_taski[label]
                for example in add_data:
                    data.append(example)

    if args.max_seq_length is None:
        max_length = tokenizer.max_len
    else:
        max_length = args.max_seq_length

    batch_encoding = tokenizer(
        [example[0] for example in data],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([batch_encoding['input_ids'][i] for i in range(len(data))], dtype=torch.long)
    all_attention_mask = torch.tensor([batch_encoding['attention_mask'][i] for i in range(len(data))], dtype=torch.long)
    all_token_type_ids = torch.tensor([batch_encoding['token_type_ids'][i] for i in range(len(data))], dtype=torch.long)

    all_labels = torch.tensor([int(example[2]) for example in data], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def calculate_loss_and_accuracy_(outputs, labels, device):
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., 1:-1, :].contiguous()
    shift_labels = labels[..., 2:].contiguous().to(device)

    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    loss = loss.view(-1, shift_logits.shape[1])

    # _, preds = shift_logits.max(dim=-1)
    not_ignore = shift_labels.ne(tokenizer.pad_token_id)
    # num_targets = not_ignore.long().sum().item()
    #
    # correct = (shift_labels == preds) & not_ignore
    # correct = correct.float().sum()
    #
    # accuracy = correct / num_targets
    # loss = loss / num_targets

    # rouge_score = rouge(not_ignore, shift_labels, preds)
    return loss, not_ignore


def prompt_contrast_train(args, model, train_dataset):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    num_training_steps = args.epochs * len(train_dataloader)
    # Prepare optimizer and schedule (linear warmup and decay)
    for param in model.transformer.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False

    model.transformer.wte.learned_embedding.requires_grad = True

    optimizer = AdamW([model.transformer.wte.learned_embedding], lr=args.lr)
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
        for batch in train_dataloader:
            batch_steps += 1

            batch = tuple(t.to(device) for t in batch)
            batch_0 = batch[0]
            pt_id = tokenizer.unk_token_id
            nt_id = tokenizer.mask_token_id

            # prepending tokens corresponding to 'positive' and 'negative' to the inputs
            pt_token = (torch.ones(batch_0.shape[0]) * pt_id).type_as(batch_0).view(-1, 1)
            nt_token = (torch.ones(batch_0.shape[0]) * nt_id).type_as(batch_0).view(-1, 1)

            seq_a = torch.cat((pt_token, batch_0), 1)
            seq_b = torch.cat((nt_token, batch_0), 1)

            bsz = seq_a.shape[0]

            # want to compute LM loss here so feeding inputs as labels
            inputs_pos = {"input_ids": seq_a, "labels": seq_a, }
            inputs_neg = {"input_ids": seq_b, "labels": seq_b, }

            outputs_a = model(**inputs_neg)  # modeling_gpt2.py modified to have none reduction

            loss_a, loss_mask = calculate_loss_and_accuracy_(outputs_a, seq_a, device)

            loss_lengths = torch.sum(loss_mask, 1, keepdim=True)

            # loss_a = outputs_a[0].view(bsz, -1)
            # loss mask includes first padded token

            outputs_b = model(**inputs_pos)

            loss_b, _ = calculate_loss_and_accuracy_(outputs_b, seq_b, device)

            # loss_b = outputs_b[0].view(bsz, -1)

            gen_loss_a = (batch[3] == 0).to(torch.float32).unsqueeze(1) * loss_a / loss_lengths
            gen_loss_b = (batch[3] == 1).to(torch.float32).unsqueeze(1) * loss_b / loss_lengths

            gen_loss = torch.sum(gen_loss_a + gen_loss_b) / bsz

            if args.sum_loss:
                loss_a = loss_a.sum(dim=1)
                loss_b = loss_b.sum(dim=1)

            else:
                loss_a = (loss_a / loss_lengths).sum(dim=1)
                loss_b = (loss_b / loss_lengths).sum(dim=1)

            class_logits = torch.stack((-loss_a, -loss_b), dim=1)  # (bsz, 2) dimensional
            class_labels = batch[3]

            if args.logit_scale:
                if not isinstance(model, torch.nn.DataParallel) and not isinstance(model,
                                                                                   torch.nn.parallel.DistributedDataParallel):
                    class_logits *= model.logit_scale
                else:
                    class_logits *= model.module.logit_scale

            if args.outbias:
                if not isinstance(model, torch.nn.DataParallel) and not isinstance(model,
                                                                                   torch.nn.parallel.DistributedDataParallel):
                    class_logits += model.bias
                else:
                    class_logits += model.module.bias

            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fn(class_logits, class_labels) * (1 - args.gen_weight) + args.gen_weight * gen_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if batch_steps % args.log_step == 0:
                print("train epoch {}/{}, batch {}/{}, loss {}".format(
                    epoch, args.epochs,
                    batch_steps,
                    num_training_steps,
                    loss
                ))

            epoch_loss_list.append(loss.cpu().detach().numpy())

            if batch_steps % 1000 == 0 and batch_steps <= 100000:
                # Save model checkpoint
                output_dir = os.path.join(args.save_model_path, "checkpoint-{}".format(batch_steps))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                model_to_save.save_pretrained(os.path.join(output_dir, "training_args.bin"))

        epoch_loss = np.mean(epoch_loss_list)
        early_stopping(epoch_loss, model, args.save_model_path)




def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="", type=str, help='')
    parser.add_argument('--vocab_path', default="", type=str, help='')
    parser.add_argument('--save_model_path', default="prompt_model", type=str, help='')
    parser.add_argument('--final_model_path', default="final_prompt_model", type=str, help='')
    parser.add_argument('--train_raw_path', default='train_raw_data.txt', type=str, help='')
    parser.add_argument('--eval_raw_path', default='test_raw_data.txt', type=str, help='')
    parser.add_argument('--batch_size', default=256, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=10001, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=500, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=1e-4, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='print log steps')
    parser.add_argument('--n_tokens', default=10, type=int, required=False, help='print log steps')
    parser.add_argument("--max_seq_length", default=34, type=int)

    # new generative classifier specific parameters
    parser.add_argument('--sup_data_num', default=0, type=int, help='the number of supervised data for each prefix')
    parser.add_argument("--balanced", action="store_true", help="use balanced dataset for training")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout prob")
    parser.add_argument("--gen_weight", default=0.8, type=float, help="scalar multiple for generative loss (lambda)")

    parser.add_argument("--logit_scale", action="store_true", help="learns to scale logits for classification")
    parser.add_argument("--threeway", action="store_true", help="does 3-way classification")
    parser.add_argument("--sum_loss", action="store_true", help="sums losses")
    parser.add_argument("--outbias", action="store_true", help="learns output bias for each class")

    return parser.parse_args()


if __name__ == '__main__':
    args = setup_args()
    args.model_path, args.vocab_path = '', '../hugging_face_test/my_token/vocab.txt'
    args.train_raw_path = '../data/contrast_data/contrast_training_data.csv'

    # args.batch_size = 16
    initialize_from_vocab = False

    tokenizer = BertTokenizer(vocab_file=args.vocab_path)

    model = GPT2LMHeadModel.from_pretrained('../hugging_face_test/best_save_model')


    s_wte = SoftEmbedding(model.get_input_embeddings(),
                          n_tokens=args.n_tokens,
                          initialize_from_vocab=initialize_from_vocab)

    model.set_input_embeddings(s_wte)

    train_dataloader = load_and_cache_examples(args, args.train_raw_path, tokenizer=tokenizer)

    prompt_contrast_train(args, model, train_dataloader)
