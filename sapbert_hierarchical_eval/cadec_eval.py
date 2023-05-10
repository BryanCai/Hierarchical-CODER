from gensim import models
import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig

import sys
sys.path.insert(1, "/home/tc24/BryanWork/CODER/pretrain")
from model import UMLSPretrainedModel


batch_size = 64
device = "cuda:0"


def cadec_eval(model, tokenizer, embed_fun):

    top_k = 3
    return eval(model, tokenizer, embed_fun, './data/cadec', top_k=top_k, summary_method="CLS") + eval(model, tokenizer, embed_fun, './data/psytar_disjoint_folds', top_k=top_k, summary_method="CLS")
    # eval(model, tokenizer, './data/cadec', top_k=top_k, summary_method="MEAN")
    
    # eval(model, tokenizer, './data/psytar_disjoint_folds', top_k=top_k, summary_method="MEAN")



def eval_one(m, tok, embed_fun, folder, top_k, summary_method=None):
    with open(os.path.join(folder, "standard.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    label2id = {line.strip().split(
        "\t")[0]: index for index, line in enumerate(lines)}
    standard_lines = [line.strip().split("\t") for line in lines]
    standard_feat = embed_fun(
        [text for (label, text) in standard_lines], m, tok, device, normalize=True, summary_method=summary_method)

    with open(os.path.join(folder, "test.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    test_lines = [line.strip().split("\t") for line in lines]
    test_feat = embed_fun(
        [text for (label, text) in test_lines], m, tok, device, normalize=True, summary_method=summary_method)

    sim_mat = np.dot(test_feat, standard_feat.T)

    correct_1 = 0
    correct_k = 0
    pred_top_k = torch.topk(torch.FloatTensor(sim_mat), k=top_k)[
        1].cpu().numpy()
    for i in range(len(test_lines)):
        true_id = label2id[test_lines[i][0]]
        if pred_top_k[i][0] == true_id:
            correct_1 += 1
        if true_id in list(pred_top_k[i]):
            correct_k += 1
    acc_1 = correct_1 / len(test_lines)
    acc_k = correct_k / len(test_lines)
    return acc_1, acc_k


def eval(m, tok, embed_fun, task_name, top_k=3, summary_method=None):
    acc_1_list = []
    acc_k_list = []
    for p in os.listdir(task_name):
        acc_1, acc_k = eval_one(m, tok, embed_fun, os.path.join(task_name, p), top_k, summary_method=summary_method)
        acc_1_list.append(acc_1)
        acc_k_list.append(acc_k)
    # print(task_name, summary_method)
    # print(f"top_k={top_k}")
    # print(acc_1_list)
    # print(acc_k_list)
    # print(sum(acc_1_list) / 5, sum(acc_k_list) / 5)

    return (sum(acc_1_list) / 5, sum(acc_k_list) / 5)


def load_vectors(filename):
    W = {}
    with open(filename, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            toks = line.strip().split()
            w = toks[0]
            vec = np.array(map(float, toks[1:]))
            W[w] = vec
    return W


def load_vectors_bin(filename):
    w = models.KeyedVectors.load_word2vec_format(filename, binary=True)
    return w


def cosine(u, v):
    return np.dot(u, v)


def norm(v):
    return np.dot(v, v)**0.5


def embed_one(phrase, dim, W):
    words = phrase.split()
    vectors = [W[w] for w in words if (w in W)]
    v = sum(vectors, np.zeros(dim))
    return v / (norm(v) + 1e-9)


def embed(phrase_list, dim, W):
    return np.array([embed_one(phrase, dim, W) for phrase in phrase_list])


if __name__ == '__main__':
    main()
