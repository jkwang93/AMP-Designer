import gzip
import pickle
import time
import torch
import argparse
import numpy as np

from transformers import BertTokenizer
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from utils import calculate_likelihood_loss, top_k_top_p_filtering, unique, decode, prompt_model_loader, \
    Variable, NLLLoss
from macrel_predictor.predictor import macrel_predictor
from soft_prompt_embedding import SoftEmbedding

import warnings
warnings.filterwarnings("ignore")
def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int, required=False, help='batch size')
    parser.add_argument('--lr', default=1e-5, type=float, required=False, help='learn rate')
    # parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--top_k', default=10, type=int, required=False, help='top k sampling')
    parser.add_argument('--top_p', default=1.0, type=float, required=False, help='')
    parser.add_argument('--sigma', default=300, type=float, required=False, help='the weight of score')
    parser.add_argument('--n_steps', default=5000, type=float, required=False, help='the weight of score')
    parser.add_argument('--n_tokens', default=10, type=int, required=False, help='soft embedding')
    return parser.parse_args()


def get_score(macrel_model,predicted_list,seqs_len):  # predicted will be the list of predicted token
    # seq_list = []
    # for index, eachseq in enumerate(predicted_list):
    #     seq = "".join(list(eachseq)).replace('[SEP]', '')
    #     seq = seq.upper()
    #     seq.replace('  ', '')
    #     seq_list.append(seq)
    '''if len samll than 6, make reward=0'''
    reward_bool = (np.array(seqs_len)>3)
    try:
        reward = macrel_predictor(predicted_list, macrel_model)
        reward = reward * reward_bool
    except Exception as e:
        reward = np.zeros(len(reward_bool))

    return reward


def likelihood(args, model, batch, tokenizer):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    start_token = ''
    start_token = tokenizer.encode(start_token)
    prompt_tokens = list(range(100, 100 + 10))
    input_each = prompt_tokens + [start_token[0]]

    # end_each_batch = torch.Tensor([start_token[1]]).long().repeat(len(batch), 1).to(device)

    input_each_batch = torch.Tensor(input_each).long().repeat(len(batch), 1).to(device)

    input_tensor = torch.cat([input_each_batch, batch], dim=1)

    inputs = {"input_ids": input_tensor.to(device)}
    outputs = model(**inputs)
    loss = calculate_likelihood_loss(outputs, input_tensor, device, tokenizer, n_tokens=args.n_tokens)
    return loss


def sample(args, model, tokenizer, batch_size, text=""):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model, _ = load_model(args.save_model_path, args.vocab_path)

    model.to(device)
    model.eval()
    time1 = time.time()
    max_length = 34

    input_ids = list(range(100, 100 + 10))

    input_ids.extend(tokenizer.encode(text))

    input_ids = input_ids[:11]

    input_tensor = torch.zeros(batch_size, 11).long()

    for index, i in enumerate(input_ids):
        input_tensor[:, index] = input_ids[index]

    Seq_list = []

    finished = torch.zeros(batch_size, 1).byte().to(device)

    log_probs = Variable(torch.zeros(batch_size))
    sequences = []
    for i in range(max_length):
        # input_tensor = torch.tensor([input_ids])
        inputs = {"input_ids": input_tensor.to(device)}
        try:
            outputs = model(**inputs)
        except Exception as e:
            print(e)

        logits = outputs.logits

        # logits = top_k_top_p_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        prob = F.softmax(logits[:, -1, :])

        log_prob = F.log_softmax(logits[:, -1, :], dim=1)

        last_token_id = torch.multinomial(prob, 1)
        sequences.append(last_token_id.view(-1, 1))
        # Flatten the tokens
        # loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
        # loss_fct = torch.nn.NLLLoss(ignore_index=tokenizer.pad_token_id, reduction="none")
        #
        # loss = loss_fct(log_prob.view(-1, log_prob.size(-1)), last_token_id.view(-1))
        loss = NLLLoss(log_prob.view(-1, log_prob.size(-1)), last_token_id.view(-1))

        log_probs += loss

        # .detach().to('cpu').numpy()
        EOS_sampled = (last_token_id == tokenizer.sep_token_id)
        finished = torch.ge(finished + EOS_sampled, 1)
        if torch.prod(finished) == 1:
            # print('End')
            break

        # last_token = tokenizer.convert_ids_to_tokens(last_token_id)
        input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
        # Seq_list.append(last_token)
    sequences = torch.cat(sequences, 1)
    # sequences = sequences.detach()
    # Seq_list = np.array(Seq_list).T
    return sequences.data, log_probs



if __name__ == '__main__':
    # set_seed(42)
    start_time = time.time()

    '''predictor'''
    macrel_model_path = "./macrel_predictor/data/models/AMP.pkl.gz"
    macrel_model = pickle.load(gzip.open(macrel_model_path, 'rb'))

    args = setup_args()
    n_steps = args.n_steps
    args.model_bin_path, args.vocab_path = '../soft_prompt/random_prompt_n_tokens/pytorch_model.bin', '../hugging_face_test/my_token/vocab.txt'
    args.model_path = '../soft_prompt/random_prompt_n_tokens'

    tokenizer = BertTokenizer(vocab_file=args.vocab_path)

    Prior = prompt_model_loader(args.model_bin_path, args.model_path)
    Agent = prompt_model_loader(args.model_bin_path, args.model_path)

    # We dont need gradients with respect to Prior
    for param in Prior.parameters():
        param.requires_grad = False

    # Agent.transformer.wte.learned_embedding.requires_grad = False
    optimizer = torch.optim.Adam(Agent.parameters(), lr=args.lr)

    for step in range(n_steps):
        # Sample from Agent
        seqs, agent_likelihood = sample(args, Agent, tokenizer, batch_size=args.batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood = likelihood(args, Prior, Variable(seqs), tokenizer)
        peptides,seqs_len = decode(seqs, tokenizer)

        score = get_score(macrel_model,peptides, seqs_len)
        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + args.sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        prior_likelihood = prior_likelihood.data.cpu().numpy()

        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Average score: {:6.2f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
            step, np.mean(score), time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             Peptides")
        for i in range(10):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       score[i],
                                                                       peptides[i]))

    # output = []
    # Seq_all = []
    # for i in range(160):
    #     print(i)
    #     Seq_list = sample(args, model, tokenizer, batch_size=16)
    #
    #     Seq_all.extend(Seq_list)
    # for j in Seq_all:
    #     output.append(decode(j))
    #
    # output = pd.DataFrame(output)
    #
    # output.to_csv('random_prompt_n_tokens.csv', index=False, header=False, sep=' ')
