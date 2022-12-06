import argparse
import pandas as pd
import torch
import numpy as np
import random
import json
import sys
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, auc
# sys.path.append('/home/tc24/BryanWork/CODER/coder_base/')
sys.path.append('/home/tc24/BryanWork/CODER/unified/')
from itertools import combinations
import time

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


def get_bert_embed(phrase_list, model, tokenizer, device, show_progress=False, batch_size = 64, summary_method="CLS", normalize=True):
    model = model.to(device)
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(tokenizer.encode_plus(
            phrase, max_length=32, add_special_tokens=True,
            truncation=True, padding="max_length")['input_ids'])
        # print(len(input_ids))
    model.eval()

    count = len(input_ids)
    now_count = 0
    output_list = []
    with torch.no_grad():
        if show_progress:
            pbar = tqdm(total=count)
        while now_count < count:
            input_gpu_0 = torch.LongTensor(input_ids[now_count:min(
                now_count + batch_size, count)]).to(device)
            if summary_method == "CLS":
                embed = model(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(model(input_gpu_0)[0], dim=1)
            if normalize:
                embed_norm = torch.norm(
                    embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                embed = embed / embed_norm
            if now_count % 1000000 == 0:
                if now_count != 0:
                    output_list.append(output.cpu().numpy())
                    del output
                output = embed
            else:
                output = torch.cat((output, embed), dim=0)
            if show_progress:
                pbar.update(min(now_count + batch_size, count) - now_count)
            now_count = min(now_count + batch_size, count)
            del input_gpu_0
        if show_progress:
            pbar.close()
    output_list.append(output.cpu().numpy())
    del output
    return np.concatenate(output_list, axis=0)

def get_sapbert_embed(phrase_list, model, tokenizer, device, show_progress=False, batch_size=2048, summary_method="CLS", normalize=True):
    model = model.to(device)
    model.eval()
    
    batch_size=batch_size
    dense_embeds = []

    with torch.no_grad():
        if show_progress:
            iterations = tqdm(range(0, len(phrase_list), batch_size))
        else:
            iterations = range(0, len(phrase_list), batch_size)
            
        for start in iterations:
            end = min(start + batch_size, len(phrase_list))
            batch = phrase_list[start:end]
            batch_tokenized_names = tokenizer.batch_encode_plus(
                    batch, add_special_tokens=True, 
                    truncation=True, max_length=32, 
                    padding="max_length", return_tensors='pt')
            batch_tokenized_names_cuda = {}
            for k,v in batch_tokenized_names.items(): 
                batch_tokenized_names_cuda[k] = v.cuda()
            
            if summary_method == "CLS":
                batch_dense_embeds = model(**batch_tokenized_names_cuda)[0][:,0,:] # [CLS]
            elif summary_method == "MEAN":
                batch_dense_embeds = model(**batch_tokenized_names_cuda)[0].mean(1) # pooling
            else:
                print ("no such summary_method:", summary_method)
            if normalize:
                embed_norm = torch.norm(
                    batch_dense_embeds, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                batch_dense_embeds = batch_dense_embeds / embed_norm

            batch_dense_embeds = batch_dense_embeds.cpu().detach().numpy()

            dense_embeds.append(batch_dense_embeds)
    dense_embeds = np.concatenate(dense_embeds, axis=0)
    
    return dense_embeds



def load_model(model_name_or_path, device):
    if model_name_or_path[-8:] == 'bert.pth':
        model = torch.load(model_name_or_path, map_location=torch.device(device))
    elif model_name_or_path[-4:] == '.pth':
        model = torch.load(model_name_or_path, map_location=torch.device(device)).bert
    else:
        model = AutoModel.from_pretrained(model_name_or_path)
    return model



def read_rank_csv(path):
    a = pd.read_csv(path)
    d = {"score": a.Mean.tolist(),
         "string1": a.Eng_Term1.tolist(),
         "string2": a.Eng_Term2.tolist()}
    return d



def read_relation_pairs(path):
    a = pd.read_csv(path, dtype=str).dropna(subset=["STR1", "STR2"])
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
                tree_data[a["tree1"].iloc[i]][a["num1"].iloc[i]].append(a["STR1"].iloc[i])
            else:
                tree_data[a["tree1"].iloc[i]][a["num1"].iloc[i]] = [a["STR1"].iloc[i]]

        if a["tree2"].iloc[i] in ['LOINC', 'PheCode', 'RXNORM', 'CCS']:
            if a["num2"].iloc[i] in tree_data[a["tree2"].iloc[i]]:
                tree_data[a["tree2"].iloc[i]][a["num2"].iloc[i]].append(a["STR2"].iloc[i])
            else:
                tree_data[a["tree2"].iloc[i]][a["num2"].iloc[i]] = [a["STR2"].iloc[i]]

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


def get_cos_sim(embed_fun, string_list1, string_list2, model, tokenizer, device):
    embed1 = embed_fun(string_list1, model, tokenizer, device, normalize=True)
    embed2 = embed_fun(string_list2, model, tokenizer, device, normalize=True)
    return [np.dot(a, b) for a, b in zip(embed1, embed2)]



def run_many(model_name_or_path, tokenizer, output_path, data_dir, device, random_samples):
    data_dir = Path(data_dir)
    model = load_model(model_name_or_path, device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    if model_name_or_path.find('SapBERT') > 0:
        embed_fun = get_sapbert_embed
    else:
        embed_fun = get_bert_embed


    output = {}
    for f in ["Similar.csv", "Relate.csv"]:
        data = read_rank_csv(data_dir/f)
        cos_sim = get_cos_sim(embed_fun, data["string1"], data["string2"], model, tokenizer, device)

        output[str(f)] = spearmanr(cos_sim, data["score"])[0]
        print(f, output[str(f)])

    pair_data, tree_terms, tree_data = read_relation_pairs(data_dir/"AllRelationPairs.csv")


    x = pd.DataFrame(combinations(tree_data["PheCode"].keys(), 2), columns=["code1", "code2"])
    x["dist"] = x.apply(lambda row: dist_PheCode(row["code1"], row["code2"]), axis=1)

    x = pd.concat([x[x["dist"] == 1], x[x["dist"] == 2], x[x["dist"] == 3].sample(100000)])
    x["term1"] = x.apply(lambda row: random.choice(tree_data["PheCode"][row["code1"]]), axis=1)
    x["term2"] = x.apply(lambda row: random.choice(tree_data["PheCode"][row["code2"]]), axis=1)

    x["cos_sim"] = get_cos_sim(embed_fun, x["term1"], x["term2"], model, tokenizer, args.device)

    for case in [(1, 2), (1, 3), (2, 3)]:
        case_label = [1]*sum(x["dist"] == case[0]) + [0]*sum(x["dist"] == case[1])
        case_sim = x[x["dist"] == case[0]]["cos_sim"].tolist() + x[x["dist"] == case[1]]["cos_sim"].tolist()

        fpr, tpr, thresholds = roc_curve(case_label, case_sim)
        auc_score = auc(fpr, tpr)
        output[str(case)] = auc_score
        print(case, output[str(case)])


    for i in pair_data:
        cos_sim = get_cos_sim(embed_fun, pair_data[i]["string1"], pair_data[i]["string2"], model, tokenizer, device)
        tree1, tree2 = i[0].split("-")
        random_terms1 = random.choices(tree_terms[tree1], k=random_samples)
        random_terms2 = random.choices(tree_terms[tree2], k=random_samples)

        random_cos_sim = get_cos_sim(embed_fun, random_terms1, random_terms2, model, tokenizer, device)

        label = [1]*len(cos_sim) + [0]*len(random_cos_sim)

        fpr, tpr, thresholds = roc_curve(label, cos_sim + random_cos_sim)
        output[str(i)] = auc(fpr, tpr)

        print(i, output[str(i)])

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
        "--model_name_or_path",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        type=str,
        help="model path",
    )

    parser.add_argument(
        "--tokenizer",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        type=str,
        help="tokenizer path",
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

    parser.add_argument(
        "--output_path",
        default="./output.json",
        type=str,
        help="output file path",
    )

    args = parser.parse_args()



    # run(args)

    model_name_or_path_list = [
                               "/home/tc24/BryanWork/saved_models/output_coder_base/model_300000.pth",
                               "/home/tc24/BryanWork/saved_models/output_unified_ms/model_300000.pth",
                               "/home/tc24/BryanWork/saved_models/old/output_unified_3/model_300000.pth",
                               "/home/tc24/BryanWork/saved_models/old/output_unified_ft_5/model_20000.pth",
                               "/home/tc24/BryanWork/saved_models/output_unified_ft_7/model_10000.pth",
                               ]
    tokenizer_list = [
                      "monologg/biobert_v1.1_pubmed",
                      "monologg/biobert_v1.1_pubmed",
                      "monologg/biobert_v1.1_pubmed",
                      "monologg/biobert_v1.1_pubmed",
                      "monologg/biobert_v1.1_pubmed",
                      ]
    output_path_list = [
                        "/home/tc24/BryanWork/saved_models/output_coder_base/output1_300000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_ms/output1_300000.json",
                        "/home/tc24/BryanWork/saved_models/old/output_unified_3/output1_300000.json",
                        "/home/tc24/BryanWork/saved_models/old/output_unified_ft_5/output1_20000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_ft_7/output1_10000.json",
                        ]

    for i in range(len(model_name_or_path_list)):
        m = model_name_or_path_list[i]
        t = tokenizer_list[i]
        o = output_path_list[i]
        print(m)
        run_many(m, t, o, args.data_dir, args.device, args.random_samples)
