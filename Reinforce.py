#!/usr/bin/env python
import argparse
import gzip
import pickle
import warnings

import torch

import numpy as np
import pandas as pd
import time

from Bio.SeqUtils import ProtParam
from transformers import BertTokenizer

import Predict
from Distillation.utils import decode, compute_peptides_similarity
from macrel_predictor.predictor import macrel_predictor
from reinforce_model_rnn import RNN
from MCMG_utils.data_structs import Vocabulary, Experience
from utils import Variable, unique

warnings.filterwarnings("ignore")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def compute_charge(seq_list):
    try:
        charge_list = []
        for peptide_seq in seq_list:
            len_peptide = len(peptide_seq)
            params = ProtParam.ProteinAnalysis(peptide_seq)
            peptide_charge = round(params.charge_at_pH(pH=7.4), 2)
            charge_list.append(peptide_charge/len_peptide)
        return charge_list
    except Exception as e:
        return 0

def get_similarity(seq_list1, seq_list2):
    try:
        sim_list = []
        for i in seq_list1:
            single_sim_list = []
            for j in seq_list2:
                sim_score = compute_peptides_similarity(i, j)
                single_sim_list.append(sim_score)
            sim_list.append(max(single_sim_list))
        return 1-np.mean(sim_list)
    except Exception as e:
        return 0

def get_score_bins(Saure_predict, Ecoli_predict, Paer_predict, macrel_model, predicted_list,
                   seqs_len, memory):  # predicted will be the list of predicted token
    '''if len samll than 6, make reward=0'''
    bins = np.linspace(0, 1, 11) 

    regression_bins = np.array([0, 10, 15, 20, 30, 50, 60, 100, 150, 250, 8156])
    regression_bins = np.log10(regression_bins)
    regression_bins[0] = -100

    try:
        macrel_reward = macrel_predictor(predicted_list, macrel_model)
        Saure_predict_regression_reward = Saure_predict.lstm_preidct(predicted_list)
        Ecoli_predict_regression_reward = Ecoli_predict.lstm_preidct(predicted_list)
        Paer_predict_regression_reward = Paer_predict.lstm_preidct(predicted_list)

        # digitize
        macrel_reward_bin = np.digitize(macrel_reward, bins)  # 返回值为每个值所属区间的索引。
        Saure_predict_regression_reward_bin = 10 - np.digitize(Saure_predict_regression_reward, regression_bins)
        Ecoli_predict_regression_reward_bin = 10 - np.digitize(Ecoli_predict_regression_reward, regression_bins)
        Paer_predict_regression_bin = 10 - np.digitize(Paer_predict_regression_reward, regression_bins)

        # similarity
        charge = compute_charge(predicted_list)

        reward = macrel_reward_bin * 0.1 + (
                Saure_predict_regression_reward_bin + Ecoli_predict_regression_reward_bin + Paer_predict_regression_bin) * 0.1 + np.mean(charge)
   
    except Exception as e:
        print(e)
        reward = np.zeros(len(seqs_len))
        macrel_reward = np.zeros(len(seqs_len))
        Saure_predict_regression_reward = np.zeros(len(seqs_len))
        Ecoli_predict_regression_reward = np.zeros(len(seqs_len))
        Paer_predict_regression_reward = np.zeros(len(seqs_len))

    return reward, macrel_reward, Saure_predict_regression_reward, Ecoli_predict_regression_reward, Paer_predict_regression_reward,charge


def get_score(predict, macrel_model, predicted_list, seqs_len):  # predicted will be the list of predicted token
    len_alpha = 0.05
    try:
        macrel_reward = macrel_predictor(predicted_list, macrel_model)
        regression_reward = predict.lstm_preidct(predicted_list)

        reward = macrel_reward - regression_reward - len_alpha * np.array(seqs_len)

    except Exception as e:
        print(e)
        reward = np.zeros(len(seqs_len))
    return reward


def train_agent(restore_prior_from='./ckpt/best_checkpoint.pt',
                restore_agent_from='./ckpt/best_checkpoint.pt', agent_save='./',
                batch_size=64, n_steps=5000, sigma=60, save_dir='./',
                experience_replay=1):
    '''predictor'''
    macrel_model_path = "AMP.pkl.gz"
    macrel_model = pickle.load(gzip.open(macrel_model_path, 'rb'))

    voc = BertTokenizer('vocab.txt')

    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(restore_prior_from, map_location={'cuda:0': 'cuda:0'}))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Prior.rnn.load_state_dict(torch.load(restore_prior_from, map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=0.0001)

    experience = Experience(voc)

    print("Model initialized, starting training...")

    # Scoring_function
    Saure_predict = Predict("")
    Ecoli_predict = Predict("")
    Paer_predict = Predict("")

    for step in range(n_steps):
        save_peptides = []
        save_score = []
        Ecoli_predict_regression_list = []
        Saure_predict_regression_list = []
        Paer_predict_regression_list = []
        charge_reward_list = []

        classification_score = []

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size=batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood = Prior.likelihood(Variable(seqs))
        peptides, seqs_len = decode(seqs, voc)

        save_peptides.extend(peptides)

        memory = experience.output_memory()
        score, classification_reward, Saure_predict_regression_reward, Ecoli_predict_regression_reward, Paer_predict_regression_reward,charge_reward = get_score_bins(
            Saure_predict, Ecoli_predict, Paer_predict, macrel_model, peptides, seqs_len, memory)

        charge_reward_list.extend(charge_reward)
        save_score.extend(score)
        Saure_predict_regression_list.extend(Saure_predict_regression_reward)
        Ecoli_predict_regression_list.extend(Ecoli_predict_regression_reward)
        Paer_predict_regression_list.extend(Paer_predict_regression_reward)
        classification_score.extend(classification_reward)

        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)

        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience) > 4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)
        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(peptides, score, prior_likelihood)
        experience.add_experience(new_experience)

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
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        if step % 100 == 0 and step != 0:
            torch.save(Agent.rnn.state_dict(), agent_save)
        print(loss)
        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Mean score: {:6.2f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
            step, np.mean(score), time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        save_score = pd.DataFrame(save_score)
        save_peptides = pd.DataFrame(save_peptides)

        classification_score = pd.DataFrame(classification_score)
        charge_reward_list = pd.DataFrame(charge_reward_list)
        Saure_predict_regression_list = pd.DataFrame(Saure_predict_regression_list)
        Ecoli_predict_regression_list = pd.DataFrame(Ecoli_predict_regression_list)
        Paer_predict_regression_list = pd.DataFrame(Paer_predict_regression_list)

        pd.concat([save_peptides, classification_score, Saure_predict_regression_list, Ecoli_predict_regression_list,
                   Paer_predict_regression_list, charge_reward_list, save_score], axis=1).to_csv(
            'generated_data/RL_sequence_step_figure.csv', index=False, header=False, mode='a')
        for i in range(10):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       score[i],
                                                                       peptides[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--num-steps', action='store', dest='n_steps', type=int,
                        default=5000)
    parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                        default=1000)
    parser.add_argument('--sigma', action='store', dest='sigma', type=int,
                        default=300)
    # parser.add_argument('--middle', action='store', dest='restore_prior_from',
    #                     default='checkpoint.pt',
    #                     help='Path to an RNN checkpoint file to use as a Prior')
    parser.add_argument('--agent', action='store', dest='agent_save',
                        default='agent_checkpoint.pt',
                        help='Path to an RNN checkpoint file to use as a Agent.')
    parser.add_argument('--save-file-path', action='store', dest='save_dir',
                        help='Path where results and model are saved. Default is data/results/run_<datetime>.')

    arg_dict = vars(parser.parse_args())

    train_agent(**arg_dict)
