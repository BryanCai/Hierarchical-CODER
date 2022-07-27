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
sys.path.append('../sanity_check/rank_loss/')
# sys.path.append('/media/sdb1/Zengsihang/Hier_CODER/Hierarchical_CODER_new/sanity_check/mod_ms_rel')
from model import UMLSPretrainedModel
import json

class AllRelDataset():
    def __init__(self, codified_data_dir, all_relations_data_path, umls_dir):
        self.umls = UMLS(umls_path=umls_dir, only_load_dict=True)
        self.load_codified_data(codified_data_dir)
        self.all_relations = pd.read_csv(all_relations_data_path)
    
    def load_codified_data(self, codified_data_dir):
        cpt_code2string = pd.read_csv(os.path.join(codified_data_dir, 'cpt/code2string.csv'))
        loinc_code2string = pd.read_csv(os.path.join(codified_data_dir, 'loinc/code2string.csv'))
        phecode_code2string = pd.read_csv(os.path.join(codified_data_dir, 'phecode/code2string.csv'))
        rxnorm_code2string = pd.read_csv(os.path.join(codified_data_dir, 'rxnorm/code2string.csv'))
        self.cpt_code2string = {code:string for code, string in zip(cpt_code2string['CPT code'], cpt_code2string['String'])}
        self.loinc_code2string = {code:string for code, string in zip(loinc_code2string['Loinc code'], loinc_code2string['String']) if code[:2] != 'LP'}
        self.phecode_code2string = {code:string for code, string in zip(phecode_code2string['Phecode'], phecode_code2string['ICD string'])}
        self.rxnorm_code2string = {code:string for code, string in zip(rxnorm_code2string['Rxnorm code'], rxnorm_code2string['String'])}
        return None

    def construct_negative_samples(self):
        self.all_relations_with_neg = self.all_relations.copy()
        self.all_relations_with_neg['neg str1'] = ''
        self.all_relations_with_neg['neg str2'] = ''
        for i in tqdm(range(len(self.all_relations))):
            pair = self.all_relations['pair'][i]
            type = self.all_relations['type'][i]
            rela = self.all_relations['RELA'][i]
            if type == 'similarity':
                if pair == 'PheCode-PheCode':
                    neg = sample(list(self.phecode_code2string.values()), 2)
                elif pair == 'RXNORM-RXNORM':
                    neg = sample(list(self.rxnorm_code2string.values()), 2)
                elif pair == 'LAB-LAB':
                    neg = sample(list(self.loinc_code2string.values()), 2)
                else:
                    neg = ['', '']
            else:
                # type == 'related'
                if pair == 'PheCode-PheCode':
                    neg = sample(list(self.phecode_code2string.values()), 2)
                elif pair == 'PheCode-RXNORM':
                    neg = sample(list(self.phecode_code2string.values()), 1) + sample(list(self.rxnorm_code2string.values()), 1)
                elif pair == 'PheCode-CCS':
                    neg = sample(list(self.phecode_code2string.values()), 1) + sample(list(self.cpt_code2string.values()), 1)
                elif pair == 'PheCode-LAB':
                    neg = sample(list(self.phecode_code2string.values()), 1) + sample(list(self.loinc_code2string.values()), 1)
                elif pair == 'RXNORM-RXNORM':
                    neg = sample(list(self.rxnorm_code2string.values()), 2)
                elif pair == 'CUI-CUI':
                    neg = sample(list(self.umls.str2cui.keys()), 2)
                    # neg = ['', '']
                else:
                    neg = ['', '']
            self.all_relations_with_neg['neg str1'][i] = neg[0]
            self.all_relations_with_neg['neg str2'][i] = neg[1]
        return None

    def print_res(self):
        self.all_relations_with_neg.to_csv('all_relations_with_neg.csv')

class AllRelEval():
    def __init__(self, data_with_neg_path, model_name_or_path, tokenizer_name, output_path):
        self.data = pd.read_csv(data_with_neg_path)
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
        self.construct_data_dict()
        self.eval()
        self.save_res()
    
    def construct_data_dict(self):
        self.data_dict = dict()
        for i in trange(len(self.data)):
            pair = self.data['pair'][i]
            if pair == 'LOINC-LOINC':
                continue
            type = self.data['type'][i]
            rela = self.data['RELA'][i]
            if pair == 'CUI-CUI':
                key = str((type, pair, rela))
            else:
                key = str((type, pair))
            posA = str(self.data['STR1'][i])
            posB = str(self.data['STR2'][i])
            negA = str(self.data['neg str1'][i])
            negB = str(self.data['neg str2'][i])
            if posA == '' or posB == '' or negA == '' or negB == '':
                continue
            if key not in self.data_dict.keys():
                self.data_dict[key] = {'listA':[], 'listB':[], 'label':[]}
            self.data_dict[key]['listA'] += [posA, negA]
            self.data_dict[key]['listB'] += [posB, negB]
            self.data_dict[key]['label'] += [1, 0]
        return None
    
    def eval(self):
        self.res = dict()
        for key, data in tqdm(self.data_dict.items()):
            if self.model_name_or_path.find('SapBERT') < 0:
                pred_sim = similarity(data['listA'], data['listB'], self.model, self.tokenizer)
            else:
                pred_sim = similarity_sapbert(data['listA'], data['listB'], self.model, self.tokenizer)
            label = data['label']
            fpr, tpr, thresholds = roc_curve(label, pred_sim)
            auc_score = auc(fpr, tpr)
            print(key, auc_score)
            self.res[key] = str(round(auc_score, 3))
        return None
    
    def save_res(self):
        with open(self.output_path, 'w') as fp:
            json.dump(self.res, fp, indent=4)



if __name__ == '__main__':
    # eval(
    #     model_name_or_path='GanjinZero/coder_eng', 
    #     tokenizer_name='GanjinZero/coder_eng', 
    #     data_path='/media/sdb1/Zengsihang/Hier_CODER/data_processing/files/AllRelationPairs.csv',
    #     out_path='coder.png'   
    # )
    # all_rel_eval = AllRelDataset(
    #     codified_data_dir='../data/cleaned/all',
    #     all_relations_data_path='../data/AllRelationPairs.csv',
    #     umls_dir='../umls'
    # )
    # all_rel_eval.construct_negative_samples()
    # all_rel_eval.print_res()
    # evaluate = AllRelEval(
    #     data_with_neg_path='all_relations_with_neg.csv', 
    #     model_name_or_path='../output_pubmed/last_model_bryan.pth', 
    #     tokenizer_name='monologg/biobert_v1.1_pubmed', 
    #     output_path='res_bryan.json'
    # )
    # evaluate = AllRelEval(
    #     data_with_neg_path='all_relations_with_neg.csv', 
    #     model_name_or_path='GanjinZero/UMLSBert_ENG', 
    #     tokenizer_name='GanjinZero/UMLSBert_ENG', 
    #     output_path='res_coder.json'
    # )
    # evaluate = AllRelEval(
    #     data_with_neg_path='all_relations_with_neg.csv', 
    #     model_name_or_path='cambridgeltl/SapBERT-from-PubMedBERT-fulltext', 
    #     tokenizer_name='cambridgeltl/SapBERT-from-PubMedBERT-fulltext', 
    #     output_path='res_sapbert.json'
    # )    
    # evaluate = AllRelEval(
    #     data_with_neg_path='all_relations_with_neg.csv', 
    #     model_name_or_path='../output_coder/model_450000.pth', 
    #     tokenizer_name='GanjinZero/coder_eng', 
    #     output_path='res_coderinit.json'
    # )
    # evaluate = AllRelEval(
    #     data_with_neg_path='all_relations_with_neg.csv', 
    #     model_name_or_path='../output_pubmed/model_200000.pth', 
    #     tokenizer_name='monologg/biobert_v1.1_pubmed', 
    #     output_path='res_200000.json'
    # )
    # evaluate = AllRelEval(
    #     data_with_neg_path='all_relations_with_neg.csv', 
    #     model_name_or_path='../output_coder_logloss/last_model.pth', 
    #     tokenizer_name='monologg/biobert_v1.1_pubmed', 
    #     output_path='res_coder_logloss.json'
    # )
    # evaluate = AllRelEval(
    #     data_with_neg_path='all_relations_with_neg.csv', 
    #     model_name_or_path='../output_tree_with_umlsneg/model_400000.pth', 
    #     tokenizer_name='monologg/biobert_v1.1_pubmed', 
    #     output_path='res_tree_with_umlsneg_400k.json'
    # )
    # evaluate = AllRelEval(
    #     data_with_neg_path='all_relations_with_neg.csv', 
    #     model_name_or_path='../output_ori_coder_filter/last_model.pth', 
    #     tokenizer_name='monologg/biobert_v1.1_pubmed', 
    #     output_path='res_oricoder_filter_last.json'
    # )
    evaluate = AllRelEval(
        data_with_neg_path='all_relations_with_neg.csv', 
        model_name_or_path='/media/sdb1/Zengsihang/Hier_CODER/Hierarchical_CODER_new/sanity_check/rank_loss/output_22/model_20000.pth', 
        tokenizer_name='monologg/biobert_v1.1_pubmed', 
        output_path='rank_loss_22.json'
    )