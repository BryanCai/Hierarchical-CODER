from data_util import UMLSDataset, TreeDataset, UMLSHoldoutDataset, fixed_length_dataloader
from torch.utils.data import DataLoader
from model import UMLSPretrainedModel
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer, BertModel
from tqdm import tqdm, trange
import torch
from torch import nn
import time
import os
import numpy as np
import pandas as pd
import argparse
import time
import pathlib
import itertools
import sys
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0")



def eval(bert_model, dataloader):

    bert_model.eval()

    cos = nn.CosineSimilarity()

    neg_dists_all = []
    cos_dists_all = []

    iterator = tqdm(dataloader, desc="Iteration", ascii=True)
    for batch in iterator:
        anchor_ids        = batch[0].to(device)
        neg_samples_ids   = batch[1].to(device)
        neg_samples_dists = batch[2].to(device)
        embed_anchor = torch.mean(bert_model(anchor_ids)[0], dim=1)
        embed_neg_samples = torch.mean(bert_model(neg_samples_ids)[0], dim=1)
        neg_dists_all.append(neg_samples_dists.cpu().detach().numpy())
        cos_dists_all.append(cos(embed_anchor, embed_neg_samples).cpu().detach().numpy())


    neg_dists_all = np.concatenate(neg_dists_all)
    cos_dists_all = np.concatenate(cos_dists_all)

    return neg_dists_all, cos_dists_all


def get_roc_tree(neg_dists, cos_dists):
    df = pd.DataFrame({"neg_dist": neg_dists, "cos_dist": cos_dists})
    df["class"] = df["neg_dist"].isin([1, 2]).astype(int)
    print(roc_auc_score(df[df["neg_dist"].isin([1, 3])]["class"], df[df["neg_dist"].isin([1, 3])]["cos_dist"]))
    print(roc_auc_score(df[df["neg_dist"].isin([2, 3])]["class"], df[df["neg_dist"].isin([2, 3])]["cos_dist"]))


if __name__ == "__main__":
    filename = sys.argv[1]

    print(filename)
    model = torch.load(filename).to(device)
    tokenizer = model.tokenizer
    model = model.bert

    coder_filename = "GanjinZero/coder_eng"
    coder_config = AutoConfig.from_pretrained(coder_filename)
    coder_tokenizer = AutoTokenizer.from_pretrained(coder_filename)
    coder_model = AutoModel.from_pretrained(
        coder_filename,
        config=coder_config).to(device)

    sapbert_filename = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    sapbert_config = AutoConfig.from_pretrained(coder_filename)
    sapbert_tokenizer = AutoTokenizer.from_pretrained(coder_filename)
    sapbert_model = AutoModel.from_pretrained(
        sapbert_filename,
        config=sapbert_config).to(device)

    tree_dir = sys.argv[2]
    print(tree_dir)

    holdout_file = sys.argv[3]
    print(holdout_file)


    for m, t in [(model, tokenizer), (coder_model, coder_tokenizer), (sapbert_model, sapbert_tokenizer)]:
        tree_dataset = TreeDataset(tree_dir=tree_dir, tokenizer=t, max_neg_samples=32)
        tree_dataloader = fixed_length_dataloader(tree_dataset, fixed_length=256, num_workers=1)
        neg_dists, cos_dists = eval(m, tree_dataloader)
        get_roc_tree(neg_dists, cos_dists)

    # umls_holdout_dataset = UMLSHoldoutDataset(holdout_file, model_name_or_path="monologg/biobert_v1.1_pubmed")
    # umls_holdout_dataloader = DataLoader(umls_holdout_dataset, batch_size=128, shuffle=False)

    # eval(model, umls_holdout_dataloader)