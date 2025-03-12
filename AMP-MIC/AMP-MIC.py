import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, GPT2LMHeadModel
import pandas as pd


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
    label_ids = []
    input_lens_list = [len(w) for w in batch]
    max_input_len = max(input_lens_list)-1
    for btc_idx in range(len(batch)):
        input_len = len(batch[btc_idx][:-1])
        input_ids.append(batch[btc_idx][:-1])
        label_ids.append([batch[btc_idx][-1]])
        input_ids[btc_idx].extend([tokenizer.pad_token_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long),torch.tensor(label_ids,dtype=torch.float32)

def data_loader(args, train_data_path, tokenizer, shuffle,n_tokens):
    data_list = []
    train_data = pd.read_csv(train_data_path,encoding='gbk').reset_index(drop=True)


    train_data = train_data[train_data['Sequence'].str.len() <= 32]

    train_x = train_data['Sequence'].values.flatten().tolist()

    for index,data_i in tqdm(enumerate(train_x)):
        seq_list = [x for x in data_i]
        seq_list = " ".join(list(seq_list))
        combine = tokenizer.encode(seq_list)
        combine.extend([0])
        data_list.append(combine)

    dataset = MyDataset(data_list)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=collate_fn)
    return dataloader


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="", type=str, help='')
    parser.add_argument('--vocab_path', default="/vocab.txt", type=str, help='')
    parser.add_argument('--save_model_path', default="prompt_model", type=str, help='')
    parser.add_argument('--final_model_path', default="final_prompt_model", type=str, help='')
    parser.add_argument('--train_raw_path', default='train_raw_data.txt', type=str, help='')
    parser.add_argument('--eval_raw_path', default='test_raw_data.txt', type=str, help='')
    parser.add_argument('--batch_size', default=16, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=10001, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=500, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=1e-4, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='print log steps')
    parser.add_argument('--n_tokens', default=10, type=int, required=False, help='print log steps')

    return parser.parse_args()

if __name__ == '__main__':

    args = setup_args()
    args.model_path, args.vocab_path = '', '../voc/vocab.txt'

    args.train_raw_path = r''

    args.batch_size = 16
    initialize_from_vocab = False

    tokenizer = BertTokenizer(vocab_file=args.vocab_path)

    GPTmodel = GPT2LMHeadModel.from_pretrained('../best_save_model')

    train_dataloader = data_loader(args, args.train_raw_path, tokenizer=tokenizer, shuffle=False, n_tokens=args.n_tokens)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # num_labels = 2
    regressor = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )
    class CombinedModel(nn.Module):
        def __init__(self, GPTmodel, regressor):
            super(CombinedModel, self).__init__()
            self.submodel1 = GPTmodel
            self.submodel2 = regressor

        def forward(self, x):
            prediction = self.submodel1(x, output_hidden_states=True)
            hidden_states = prediction.hidden_states[-1].squeeze(0)

            pooled_output = hidden_states[:, -1, :]
            logits = regressor(pooled_output)
            return logits

    model = CombinedModel(GPTmodel, regressor)
    model.load_state_dict(torch.load('E_fine_tuning_params/regress.pth', map_location={'cuda:0': 'cuda:0'}))
    model.to(device)

    criterion = nn.MSELoss()
    predict_score = []

    model.eval()
    epoch_loss_list = []

    loss_all = 0
    count = 0

    with torch.no_grad():
        for batch in train_dataloader:
            input_ids, _ = batch
            out = model(input_ids.to(device))
            out = 10**out
            out = out.cpu().detach().numpy().tolist()
            predict_score.extend(out)
    predict_score = pd.DataFrame(predict_score)

    predict_score.to_csv('', index=False, header=[''],  mode='a')