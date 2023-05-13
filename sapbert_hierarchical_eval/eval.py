import argparse
import pandas as pd
import torch
import numpy as np
import random
import json
import sys
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, auc
from cadec_eval import cadec_eval
from load_trees import TREE
from itertools import combinations
import time
from collections import Counter
import csv
import re

from embed_functions import (
    get_bert_embed,
    get_biogpt_embed,
    get_sapbert_embed,
    get_distilbert_embed,
    get_wrapper_embed,
    get_truncated_embed_fun,
)

from loaders import (
    load_model_and_tokenizer,
    load_model_and_tokenizer_bert,
    load_model_and_tokenizer_biogpt,
    load_model_and_tokenizer_SapBERT,
    load_model_and_tokenizer_wrapper,
)


def clean(term, lower=True, clean_NOS=True, clean_bracket=True, clean_dash=True):
    term = " " + term + " "
    if lower:
        term = term.lower()
    if clean_NOS:
        term = term.replace(" NOS ", " ").replace(" nos ", " ")
    if clean_bracket:
        term = re.sub(u"\\(.*?\\)", "", term)
    if clean_dash:
        term = term.replace("-", " ")
    term = " ".join([w for w in term.split() if w])
    return term

def dist_PheCode(code1, code2):
    def split_PheCode(code):
        out = []
        if "." in code:
            out.append(code.split(".")[0])
            for i in range(len(code.split(".")[1])):
                out.append(code.split(".")[1][i])
            for i in range(2 - len(code.split(".")[1])):
                out.append(None)
        else:
            out = [code, None, None]
        return out

    if code1 == code2:
        return 0

    split1 = split_PheCode(code1)
    split2 = split_PheCode(code2)

    if split1[0] == split2[0]:
        assert not (split1[1] is None and split2[1] is None)
        if split1[1] == split2[1]:
            assert not (split1[2] is None and split2[2] is None)
            return 1
        else:
            return 2
    else:
        return 3


def read_rank_csv(path):
    a = pd.read_csv(path)
    d = {"score": a.Mean.tolist(),
         "string1": a.Eng_Term1.tolist(),
         "string2": a.Eng_Term2.tolist()}
    return d



def read_relation_pairs(data_path, tree_path):
    a = pd.read_csv(data_path, dtype=str).dropna(subset=["STR1", "STR2"])
    a.pair.replace({"LAB-LAB": "LOINC-LOINC", "PheCode-LAB": "PheCode-LOINC"}, inplace=True)
    pair_data = {}
    for pair_type in a.pair.unique():
        b = a[a.pair == pair_type]
        if pair_type != "CUI-CUI":
            for rel_type in b.type.unique():
                pair_data[(pair_type, rel_type)] = {"string1": b[b.type == rel_type].STR1.tolist(),
                                                    "string2": b[b.type == rel_type].STR2.tolist()}
        else:
            for rel_type in b.RELA.unique():
                pair_data[(pair_type, rel_type)] = {"string1": b[b.RELA == rel_type].STR1.tolist(),
                                                    "string2": b[b.RELA == rel_type].STR2.tolist()}

    a["tree1"] = a.code1.apply(lambda x: str(x).split(":")[0])
    a["tree2"] = a.code2.apply(lambda x: str(x).split(":")[0])

    a["num1"] = a.code1.apply(lambda x: "" if len(str(x).split(":")) < 2 else str(x).split(":")[1])
    a["num2"] = a.code2.apply(lambda x: "" if len(str(x).split(":")) < 2 else str(x).split(":")[1])




    tree_data = {}
    for i in ['LOINC', 'PheCode', 'RXNORM', 'CCS']:
        tree_data[i] = {}
    for i in range(a.shape[0]):
        if a["tree1"].iloc[i] in ['LOINC', 'PheCode', 'RXNORM', 'CCS']:
            if a["num1"].iloc[i] in tree_data[a["tree1"].iloc[i]]:
                tree_data[a["tree1"].iloc[i]][a["num1"].iloc[i]].add(a["STR1"].iloc[i].lower())
            else:
                tree_data[a["tree1"].iloc[i]][a["num1"].iloc[i]] = set([a["STR1"].iloc[i].lower()])

        if a["tree2"].iloc[i] in ['LOINC', 'PheCode', 'RXNORM', 'CCS']:
            if a["num2"].iloc[i] in tree_data[a["tree2"].iloc[i]]:
                tree_data[a["tree2"].iloc[i]][a["num2"].iloc[i]].add(a["STR2"].iloc[i].lower())
            else:
                tree_data[a["tree2"].iloc[i]][a["num2"].iloc[i]] = set([a["STR2"].iloc[i].lower()])


    phecode_term_data = {}
    with open(tree_path/"phecode"/"code2string.csv") as f:
        reader = csv.reader(f)
        reader.__next__()
        for row in reader:
            current = row[0]
            text = row[1]
            if current not in phecode_term_data:
                phecode_term_data[current] = set([clean(text)])
            phecode_term_data[current].add(clean(text))
    unknown_codes = []
    for i in tree_data["PheCode"]:
        if i not in phecode_term_data:
            unknown_codes.append(i)
        else:
            tree_data["PheCode"][i] = phecode_term_data[i]
    for i in unknown_codes:
        del tree_data["PheCode"][i]


    tree_terms = {}
    for tree in ['LOINC', 'PheCode', 'RXNORM', 'CCS']:
        x = set()
        x.update(a[a.tree1 == tree].STR1)
        x.update(a[a.tree2 == tree].STR2)
        tree_terms[tree] = list(x)

    x = set()
    x.update(a[a.CUI1.isna() == False].STR1)
    x.update(a[a.CUI2.isna() == False].STR2)
    tree_terms["CUI"] = list(x)

    return pair_data, tree_terms, tree_data


def read_phecode_eval(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()

    eval_data = []
    for l in lines:
        lst = l.rstrip("\n").split("||")
        eval_data.append([int(lst[0]), lst[1], lst[2]])

    return pd.DataFrame(eval_data, columns=["dist", "term1", "term2"])

def read_cui_cui_eval(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()

    eval_data = []
    for l in lines:
        lst = l.rstrip("\n").split("||")
        eval_data.append(lst)

    return pd.DataFrame(eval_data, columns=["term1", "term2", "relation"])


def get_cos_sim(embed_fun, string_list1, string_list2, model, tokenizer, device):
    embed1 = embed_fun(string_list1, model, tokenizer, device, normalize=True)
    embed2 = embed_fun(string_list2, model, tokenizer, device, normalize=True)
    return [np.dot(a, b) for a, b in zip(embed1, embed2)]



def run_many(model_name_or_path, util_function, output_path, data_dir, tree_dir, device, random_samples):
    data_dir = Path(data_dir)
    tree_dir = Path(tree_dir)
    model, tokenizer = util_function[0](model_name_or_path, device)
    embed_fun = util_function[1]

    output = {}
    for f in ["Similar.csv", "Relate.csv"]:
        data = read_rank_csv(data_dir/f)
        cos_sim = get_cos_sim(embed_fun, data["string1"], data["string2"], model, tokenizer, device)

        output[str(f)] = spearmanr(cos_sim, data["score"])[0]
        print(f, output[str(f)])



    x = read_phecode_eval(data_dir/"phecode_eval.txt")

    x["cos_sim"] = get_cos_sim(embed_fun, x["term1"].tolist(), x["term2"].tolist(), model, tokenizer, device)

    for case in combinations(range(4), 2):
        case_label = [1]*sum(x["dist"] == case[0]) + [0]*sum(x["dist"] == case[1])
        case_sim = x[x["dist"] == case[0]]["cos_sim"].tolist() + x[x["dist"] == case[1]]["cos_sim"].tolist()

        fpr, tpr, thresholds = roc_curve(case_label, case_sim)
        auc_score = auc(fpr, tpr)
        output[str(case)] = auc_score
        print(case, output[str(case)])

    x = read_cui_cui_eval(data_dir/"cui_cui_eval.txt")

    relation_list = [
                     'ALL_CAUSITIVE',
                     'ALL_MAY_CAUSE_OR_TREAT',
                     'method_of',
                     'classifies',
                     'DDX',
                     'component_of',
                     'classified_as',
                     'finding_site_of',
                     'translation_of',
                     'isa',
                     'subset_includes_concept',
                     'inverse_isa',
                     'has_ingredient',
                     'concept_in_subset',
                     'mapped_from',
                     'has_member',
                     'member_of',
                     'has_translation',
                     'expanded_form_of',
                     'mapped_to',
                     'has_inactive_ingredient',
                     'ingredient_of',
                     'inactive_ingredient_of',
                     'has_finding_site',
                     'classifies',
                     'active_moiety_of',
                     'has_component',
                     'has_active_ingredient',
                     'active_ingredient_of',
                     'has_active_moiety',
                     'has_class',
                     'has_expanded_form',
                     'class_of'
                     ]
                     
    x = x[x["relation"].isin(relation_list + ["random"])]

    x["cos_sim"] = get_cos_sim(embed_fun, x["term1"].tolist(), x["term2"].tolist(), model, tokenizer, device)

    for relation in relation_list:

        case_label = [1]*sum(x["relation"] == relation) + [0]*sum(x["relation"] == "random")
        case_sim = x[x["relation"] == relation]["cos_sim"].tolist() + x[x["relation"] == "random"]["cos_sim"].tolist()

        fpr, tpr, thresholds = roc_curve(case_label, case_sim)
        auc_score = auc(fpr, tpr)
        output[relation] = auc_score
        print(relation, output[relation])


    output["cadec"] = cadec_eval(model, tokenizer, embed_fun)

    example_pairs = [
                     ("Type 1 Diabetes", "Type 2 Diabetes"),
                     ("Parkinson's disease", "Lewy body dementia"),
                     ("Alzheimer's disease", "Dementia"),
                     ("Osteoarthritis", "Rheumatoid Arthritis"),
                     ("adenocarcinoma", "squamous cell carcinoma"),
                    ]

    term1_list, term2_list = map(list, zip(*example_pairs))
    example_cos_sim = get_cos_sim(embed_fun, term1_list, term2_list, model, tokenizer, device)
    for (i, sim) in enumerate(example_cos_sim):
        output["example" + str(i) + "_cos_sim"] = float(sim)
        print("example" + str(i) + "_cos_sim", output["example" + str(i) + "_cos_sim"])

    with open(output_path, 'w') as fp:
        json.dump(output, fp, indent=4)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="./data",
        type=str,
        help="eval data directory",
    )
    parser.add_argument(
        "--tree_dir",
        default="/home/tc24/BryanWork/CODER/data/cleaned/all",
        type=str,
        help="Path to tree directory",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="tokenizer path",
    )
    parser.add_argument(
        "--random_samples",
        default=10000,
        type=int,
        help="number of random samples",
    )

    args = parser.parse_args()


    model_name_or_path_list = [
                               # "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/triplet_umls_full/final",
                               # "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/triplet_umls_rela_tree_full/final",
                               # "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/triplet_umls_rela_tree_100/final",
                               # "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/triplet_umls_rela_tree_100/final",
                               # "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/clogit_01_triplet_umls_rela_tree_full_lr_6/final",
                               # "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/clogit_01_triplet_umls_rela_tree_100_lr_6/final",
                               # "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/clogit_01_triplet_umls_rela_tree_100_lr_6/final",
                               # "/home/tc24/BryanWork/saved_models/output_coder_base/model_300000.pth",
                               # "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                               # "GanjinZero/UMLSBert_ENG",
                               "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                               # "monologg/biobert_v1.1_pubmed",
                               # "microsoft/biogpt",
                               # "distilbert-base-uncased",
                               ]

    util_function_list = [
                          # (load_model_and_tokenizer_wrapper, get_wrapper_embed),
                          # (load_model_and_tokenizer_wrapper, get_wrapper_embed),
                          # (load_model_and_tokenizer_wrapper, get_truncated_embed_fun(get_wrapper_embed, 100)),
                          # (load_model_and_tokenizer_wrapper, get_wrapper_embed),
                          # (load_model_and_tokenizer_wrapper, get_wrapper_embed),
                          # (load_model_and_tokenizer_wrapper, get_truncated_embed_fun(get_wrapper_embed, 100)),
                          # (load_model_and_tokenizer_wrapper, get_wrapper_embed),
                          # (load_model_and_tokenizer_bert, get_bert_embed),
                          # (load_model_and_tokenizer_SapBERT, get_sapbert_embed),
                          # (load_model_and_tokenizer, get_bert_embed),
                          (load_model_and_tokenizer, get_bert_embed),
                          # (load_model_and_tokenizer, get_bert_embed),
                          # (load_model_and_tokenizer_biogpt, get_biogpt_embed),
                          # (load_model_and_tokenizer, get_distilbert_embed),
                          ]

    output_path_list = [
                        # "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/triplet_umls_full/output.json",
                        # "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/triplet_umls_rela_tree_full/output.json",
                        # "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/triplet_umls_rela_tree_100/output_100.json",
                        # "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/triplet_umls_rela_tree_100/output.json",
                        # "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/clogit_01_triplet_umls_rela_tree_full_lr_6/output.json",
                        # "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/clogit_01_triplet_umls_rela_tree_100_lr_6/output_100.json",
                        # "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/clogit_01_triplet_umls_rela_tree_100_lr_6/output.json",
                        # "/home/tc24/BryanWork/saved_models/output_coder_base/output.json",
                        # "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/sapbert.json",
                        # "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/coder.json",
                        "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/pubmedbert.json",
                        # "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/biobert1_1.json",
                        # "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/biogpt.json",
                        # "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/distilbert.json",
                        ]

    for i in range(len(model_name_or_path_list)):
        m = model_name_or_path_list[i]
        u = util_function_list[i]
        o = output_path_list[i]
        print(m)
        run_many(m, u, o, args.data_dir, args.tree_dir, args.device, args.random_samples)
