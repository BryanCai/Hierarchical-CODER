import csv
from tqdm import tqdm, trange
import torch
from transformers import AutoModel, AutoTokenizer
from cal_sim import similarity, similarity_sapbert
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from load_umls import UMLS
import os
from random import sample
import sys
# sys.path.append('/media/sdb1/Zengsihang/Hier_CODER/Hierarchical_CODER_new/sanity_check/simplify_ms_codified/')
# sys.path.append('/media/sdb1/Zengsihang/Hier_CODER/Hierarchical-CODER/pretrain')
# from model import UMLSPretrainedModel
import json
from scipy.stats import spearmanr

class EvalRank():
    def __init__(self, similar_path, related_path, chinese_term_pair_path, model_name_or_path, tokenizer_name, output_path):
        self.similar_score, self.similar_string1, self.similar_string2 = self.read_txt(similar_path)
        self.related_score, self.related_string1, self.related_string2 = self.read_txt(related_path)
        self.similar_score_ch, self.related_score_ch, self.string3, self.string4 = self.read_chinese_term_pairs(chinese_term_pair_path)
        self.model_name_or_path = model_name_or_path
        if model_name_or_path[-8:] == 'bert.pth':
            self.model = torch.load(model_name_or_path, map_location=torch.device('cuda:1'))
        elif model_name_or_path[-4:] == '.pth':
            self.model = torch.load(model_name_or_path, map_location=torch.device('cuda:1')).bert
            torch.save(self.model, model_name_or_path[:-4] + '_bert.pth')
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.output_path = output_path
        self.eval()
        self.save_res()

    def read_txt(self, path):
        # first row is the titles, so we skip it
        # the first element of each row is the score, save it in a list
        # the third and fourth element of each row is the string1 and string2, save them in two lists respectively
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
            scores = []
            string1s = []
            string2s = []
            for line in lines:
                line = line.strip()
                line = line.split('\t')
                scores.append(float(line[0]))
                string1s.append(line[2])
                string2s.append(line[3])
        return scores, string1s, string2s
    
    def read_chinese_term_pairs(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
            sim_scores = []
            rel_scores = []
            string1s = []
            string2s = []
            for line in lines:
                line = line.strip()
                line = line.split('\t')
                sim_scores.append(float(line[7]))
                rel_scores.append(float(line[8]))
                string1s.append(line[2])
                string2s.append(line[4])
        return sim_scores, rel_scores, string1s, string2s

    def eval(self):
        self.res = dict()
        if self.model_name_or_path.find('SapBERT') < 0:
            pred_sim = similarity(self.similar_string1, self.similar_string2, self.model, self.tokenizer)
        else:
            pred_sim = similarity_sapbert(self.similar_string1, self.similar_string2, self.model, self.tokenizer)
        label = self.similar_score
        self.res['similar1'] = spearmanr(pred_sim, label)[0]
        print('similar1: ', self.res['similar1'])

        if self.model_name_or_path.find('SapBERT') < 0:
            pred_rel = similarity(self.related_string1, self.related_string2, self.model, self.tokenizer)
        else:
            pred_rel = similarity_sapbert(self.related_string1, self.related_string2, self.model, self.tokenizer)
        label = self.related_score
        self.res['related1'] = spearmanr(pred_rel, label)[0]
        print('related1: ', self.res['related1'])

        if self.model_name_or_path.find('SapBERT') < 0:
            pred_sim = similarity(self.string3, self.string4, self.model, self.tokenizer)
        else:
            pred_sim = similarity_sapbert(self.string3, self.string4, self.model, self.tokenizer)
        label = self.similar_score_ch
        self.res['similar2'] = spearmanr(pred_sim, label)[0]
        print('similar2: ', self.res['similar2'])     

        if self.model_name_or_path.find('SapBERT') < 0:
            pred_sim = similarity(self.string3, self.string4, self.model, self.tokenizer)
        else:
            pred_sim = similarity_sapbert(self.string3, self.string4, self.model, self.tokenizer)
        label = self.related_score_ch
        self.res['related2'] = spearmanr(pred_sim, label)[0]
        print('related2: ', self.res['related2'])        
        return None

    def save_res(self):
        with open(self.output_path, 'w') as fp:
            json.dump(self.res, fp, indent=4)


if __name__ == '__main__':
    # for path in os.listdir('compare_models'):
    #     EvalRank(
    #         similar_path='similar.txt',
    #         related_path='related.txt',
    #         chinese_term_pair_path='chinesetermpairs.txt',
    #         model_name_or_path=os.path.join('compare_models/', path),
    #         tokenizer_name='monologg/biobert_v1.1_pubmed',
    #         output_path='rank_res_sanity_'+path[:-4]+'.json'
    #     )
    EvalRank(
        similar_path='similar.txt',
        related_path='related.txt',
        chinese_term_pair_path='chinesetermpairs.txt',
        model_name_or_path='/media/sdb1/Zengsihang/Hier_CODER/Hierarchical_CODER_new/sanity_check/simplify_ms_codified/output_250/model_5000_bert.pth',
        tokenizer_name='monologg/biobert_v1.1_pubmed',
        output_path='rank_res_sanity_250_mod_ms.json'
    )