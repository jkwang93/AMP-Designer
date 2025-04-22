import time
import torch
import argparse
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer
import torch.nn.functional as F

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
    parser.add_argument('--batch_size', default=128, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=1001, type=int, required=False, help='epochs')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=100, type=int, required=False, help='print log steps')
    return parser.parse_args()


def decode(matrix):
    chars = []
    for i in matrix:
        if i == '[SEP]': break
        chars.append(i.upper())
    seq = "".join(chars)
    return seq

def predict(model, tokenizer, batch_size, text=""):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    model.to(device)
    model.eval()
    time1 = time.time()
    max_length = 32
    input_ids = []
    input_ids.extend(tokenizer.encode(text))
    input_ids = input_ids[0]

    input_tensor = torch.zeros(batch_size, 1).long()

    input_tensor[:] = input_ids

    Seq_list = []

    finished = torch.zeros(batch_size,1).byte().to(device)

    for i in range(max_length):
        # input_tensor = torch.tensor([input_ids])
        inputs = {"input_ids": input_tensor.to(device)}
        outputs = model(**inputs)
        logits = outputs.logits

        logits = F.softmax(logits[:,-1,:])

        last_token_id = torch.multinomial(logits, 1)
        # .detach().to('cpu').numpy()
        EOS_sampled = (last_token_id == tokenizer.sep_token_id)
        finished = torch.ge(finished + EOS_sampled, 1)
        if torch.prod(finished) == 1:
            print('End')
            break

        last_token = tokenizer.convert_ids_to_tokens(last_token_id)

        input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)

        Seq_list.append(last_token)
    # print(Seq_list)
    Seq_list = np.array(Seq_list).T


    print("time cost: {}".format(time.time() - time1))
    return Seq_list
    # print(Seq_list)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    args = setup_args()
    args.model_path, args.vocab_path = '', './my_token/vocab.txt'


    tokenizer = BertTokenizer(vocab_file=args.vocab_path)

    model = GPT2LMHeadModel.from_pretrained(args.model_path)

    output = []
    Seq_all = []
    for i in range(100):
        Seq_list = predict(model,tokenizer,batch_size=128)

        Seq_all.extend(Seq_list)
    for j in Seq_all:
        output.append(decode(j))

    output = pd.DataFrame(output)

    output.to_csv('generate_seq.csv', index=False, header=False, sep=' ')


