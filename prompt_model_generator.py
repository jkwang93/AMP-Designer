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
from Distillation.utils import top_k_top_p_filtering
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


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="", type=str, help='')
    parser.add_argument('--vocab_path', default="", type=str, help='')
    parser.add_argument('--save_model_path', default="save_model", type=str, help='')
    parser.add_argument('--final_model_path', default="final_model", type=str, help='')
    parser.add_argument('--train_raw_path', default='train_raw_data.txt', type=str, help='')
    parser.add_argument('--eval_raw_path', default='test_raw_data.txt', type=str, help='')
    parser.add_argument('--batch_size', default=128, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=1001, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=1e-3, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=100, type=int, required=False, help='print log steps')
    return parser.parse_args()

def decode(matrix):
    chars = []
    for i in matrix:
        if i == '[SEP]':
            break
        chars.append(i.upper())
    seq = "".join(chars)
    return seq

def predict(model, tokenizer, batch_size, text=""):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    model.eval()
    time1 = time.time()
    max_length = 34

    input_ids = list(range(100, 100 + 10))

    input_ids.extend(tokenizer.encode(text))

    input_ids = input_ids[:11]

    input_tensor = torch.zeros(batch_size, 11).long()

    for index,i in enumerate(input_ids):
        input_tensor[:,index] = input_ids[index]

    Seq_list = []

    finished = torch.zeros(batch_size,1).byte().to(device)

    for i in range(max_length):
        inputs = {"input_ids": input_tensor.to(device)}
        try:
            outputs = model(**inputs)
        except Exception as e:
            print(e)

        logits = outputs.logits

        # if topk
        # logits = top_k_top_p_filtering(logits.view(batch_size, 1, -1), top_k=10, top_p=1.0)

        logits = F.softmax(logits[:,-1,:])
        last_token_id = torch.multinomial(logits, 1)
        EOS_sampled = (last_token_id == tokenizer.sep_token_id)
        finished = torch.ge(finished + EOS_sampled, 1)
        if torch.prod(finished) == 1:
            print('End')
            break

        last_token = tokenizer.convert_ids_to_tokens(last_token_id)
        input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
        Seq_list.append(last_token)
    Seq_list = np.array(Seq_list).T
    print("time cost: {}".format(time.time() - time1))
    return Seq_list


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':

    args = setup_args()
    args.model_path, args.vocab_path = '', '../my_token/vocab.txt'

    tokenizer = BertTokenizer(vocab_file=args.vocab_path)

    prompt_model_load = torch.load("random_prompt_n_tokens/pytorch_model.bin")

    model= GPT2LMHeadModel.from_pretrained('./random_prompt_n_tokens')
    s_wte = SoftEmbedding(model.get_input_embeddings(),
                          n_tokens=10,
                          initialize_from_vocab=True)

    s_wte.learned_embedding.data = prompt_model_load['transformer.wte.learned_embedding']
    s_wte.wte.weight.data = prompt_model_load['transformer.wte.wte.weight']

    del prompt_model_load
    model.set_input_embeddings(s_wte)

    output = []
    Seq_all = []
    for i in range(10):
        print(i)
        Seq_list = predict(model,tokenizer,batch_size=64)

        Seq_all.extend(Seq_list)
    for j in Seq_all:
        output.append(decode(j))

    output = pd.DataFrame(output)

    output.to_csv('prompt.csv', index=False, header=False, sep=' ')


