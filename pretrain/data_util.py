import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from load_umls import UMLS
from load_trees import TREE
from torch.utils.data import Dataset, DataLoader
import random
from sampler_util import FixedLengthBatchSampler, my_collate_fn
from torch.utils.data.sampler import RandomSampler
import ipdb
from time import time
import json


def pad(list_ids, pad_length, pad_mark=0):
    output = []
    for l in list_ids:
        if len(l) > pad_length:
            output.append(l[0:pad_length])
        else:
            output.append(l + [pad_mark] * (pad_length - len(l)))
    return output


def my_sample(lst, lst_length, start, length):
    start = start % lst_length
    if start + length < lst_length:
        return lst[start:start + length]
    return lst[start:] + lst[0:start + length - lst_length]


class UMLSDataset(Dataset):
    def __init__(self, umls_folder, model_name_or_path, lang, json_save_path=None, max_lui_per_cui=8, max_length=32):
        self.umls = UMLS(umls_folder, lang_range=lang)
        self.len = len(self.umls.rel)
        self.max_lui_per_cui = max_lui_per_cui
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.json_save_path = json_save_path
        self.calculate_class_count()

    def calculate_class_count(self):
        print("Calculate class count")

        self.cui2id = {cui: index for index,
                       cui in enumerate(self.umls.cui2str.keys())}
        self.id2cui = {v: k for k, v in self.cui2id.items()}

        self.re_set = set()
        self.rel_set = set()
        for r in self.umls.rel:
            _, _, re, rel = r.split("\t")
            self.re_set.update([re])
            self.rel_set.update([rel])
        self.re_set = list(self.re_set)
        self.rel_set = list(self.rel_set)
        self.re_set.sort()
        self.rel_set.sort()

        self.re2id = {re: index for index, re in enumerate(self.re_set)}
        self.rel2id = {rel: index for index, rel in enumerate(self.rel_set)}

        sty_list = list(set(self.umls.cui2sty.values()))
        sty_list.sort()
        self.sty2id = {sty: index for index, sty in enumerate(sty_list)}

        if self.json_save_path:
            with open(os.path.join(self.json_save_path, "re2id.json"), "w") as f:
                json.dump(self.re2id, f)
            with open(os.path.join(self.json_save_path, "rel2id.json"), "w") as f:
                json.dump(self.rel2id, f)
            with open(os.path.join(self.json_save_path, "sty2id.json"), "w") as f:
                json.dump(self.sty2id, f)

        print("CUI:", len(self.cui2id))
        print("RE:", len(self.re2id))
        print("REL:", len(self.rel2id))
        print("STY:", len(self.sty2id))

    def tokenize_one(self, string):
        return self.tokenizer.encode_plus(string, max_length=self.max_length, truncation=True)['input_ids']

    # @profile
    def __getitem__(self, index):
        cui0, cui1, re, rel = self.umls.rel[index].split("\t")

        str0_list = list(self.umls.cui2str[cui0])
        str1_list = list(self.umls.cui2str[cui1])
        if len(str0_list) > self.max_lui_per_cui:
            str0_list = random.sample(str0_list, self.max_lui_per_cui)
        if len(str1_list) > self.max_lui_per_cui:
            str1_list = random.sample(str1_list, self.max_lui_per_cui)
        use_len = min(len(str0_list), len(str1_list))
        str0_list = str0_list[0:use_len]
        str1_list = str1_list[0:use_len]

        sty0_index = self.sty2id[self.umls.cui2sty[cui0]]
        sty1_index = self.sty2id[self.umls.cui2sty[cui1]]

        str2_list = []
        cui2_index_list = []
        sty2_index_list = []

        cui2 = my_sample(self.umls.cui, self.umls.cui_count,
                         index * self.max_lui_per_cui, use_len * 2)
        sample_index = 0
        while len(str2_list) < use_len:
            if sample_index < len(cui2):
                use_cui2 = cui2[sample_index]
            else:
                sample_index = 0
                cui2 = my_sample(self.umls.cui, self.umls.cui_count,
                                 index * self.max_lui_per_cui, use_len * 2)
                use_cui2 = cui2[sample_index]
            # if not "\t".join([cui0, use_cui2, re, rel]) in self.umls.rel: # TOO SLOW!
            if True:
                cui2_index_list.append(self.cui2id[use_cui2])
                sty2_index_list.append(
                    self.sty2id[self.umls.cui2sty[use_cui2]])
                str2_list.append(random.sample(self.umls.cui2str[use_cui2], 1)[0])
                sample_index += 1

        # print(str0_list)
        # print(str1_list)
        # print(str2_list)

        input_ids = [self.tokenize_one(s)
                     for s in str0_list + str1_list + str2_list]
        input_ids = pad(input_ids, self.max_length)
        input_ids_0 = input_ids[0:use_len]
        input_ids_1 = input_ids[use_len:2 * use_len]
        input_ids_2 = input_ids[2 * use_len:]

        cui0_index = self.cui2id[cui0]
        cui1_index = self.cui2id[cui1]

        re_index = self.re2id[re]
        rel_index = self.rel2id[rel]
        return input_ids_0, input_ids_1, input_ids_2, \
            [cui0_index] * use_len, [cui1_index] * use_len, cui2_index_list, \
            [sty0_index] * use_len, [sty1_index] * use_len, sty2_index_list, \
            [re_index] * use_len, \
            [rel_index] * use_len

    def __len__(self):
        return self.len


def fixed_length_dataloader(dataset, fixed_length=96, num_workers=0):
    base_sampler = RandomSampler(dataset)
    batch_sampler = FixedLengthBatchSampler(
        sampler=base_sampler, fixed_length=fixed_length, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler,
                            collate_fn=my_collate_fn, num_workers=num_workers, pin_memory=True)
    return dataloader


class TreeDataset(Dataset):
    def __init__(self, loinc_tree_path, loinc_map_path, rxnorm_tree_path, rxnorm_map_path, cpt_tree_path, cpt_map_path, model_name_or_path, max_neg_samples=8, max_length=32):
        self.trees = {}
        self.trees['loinc'] = TREE(loinc_tree_path, loinc_map_path)
        self.trees['rxnorm'] = TREE(rxnorm_tree_path, rxnorm_map_path)
        self.trees['cpt'] = TREE(cpt_tree_path, cpt_map_path)
        self.obj_list = []
        self.len = 0
        for tree in self.trees:
            self.obj_list += [(i, tree) for i in self.trees[tree].text.keys()]
            self.len += len(self.trees[tree])
        self.max_neg_samples = max_neg_samples
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def __getitem__(self, index):
        anchor_id, tree = self.obj_list[index]
        if anchor_id not in self.trees[tree].text:
            return [], [], []
        neg_samples_close = []
        neg_samples_far = []

        neg_samples_close += [(i, 1) for i in self.trees[tree].children[anchor_id]]
        neg_samples_far += [(i, 2) for i in self.trees[tree].grandchildren[anchor_id]]
        if anchor_id in self.trees[tree].parent:
            parent = self.trees[tree].parent[anchor_id]
            neg_samples_close += [(parent, 1)]
            neg_samples_far += [(i, 2) for i in self.trees[tree].children[parent] if i != anchor_id]
            if parent in self.trees[tree].parent:
                grandparent = self.trees[tree].parent[parent]
                neg_samples_far += [(grandparent, 2)]

        neg_samples_close = [i for i in neg_samples_close if i[0] in self.trees[tree].text]
        neg_samples_far = [i for i in neg_samples_far if i[0] in self.trees[tree].text]

        if len(neg_samples_close) > self.max_neg_samples:
            neg_samples = random.sample(neg_samples_close, self.max_neg_samples)
        else:
            neg_samples = neg_samples_close

        neg_samples += random.sample(neg_samples_far, min(len(neg_samples_far), self.max_neg_samples - len(neg_samples)))

        if len(neg_samples) == 0:
            return [], [], []

        neg_samples_id, neg_samples_dist = list(zip(*neg_samples))

        anchor_string = random.choice(self.trees[tree].text[anchor_id])
        neg_samples_string = [random.choice(self.trees[tree].text[i]) for i in neg_samples_id]


        anchor_input_id = self.tokenize_one(anchor_string)
        neg_samples_input_id = [self.tokenize_one(s) for s in neg_samples_string]

        return [anchor_input_id]*len(neg_samples_input_id), neg_samples_input_id, neg_samples_dist

    def __len__(self):
        return self.len


    def tokenize_one(self, string):
        return self.tokenizer.encode_plus(string, max_length=self.max_length, truncation=True, padding='max_length')['input_ids']


if __name__ == "__main__":
    loinc_tree_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/loinc/loinc_hierarchy.csv"
    loinc_map_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/loinc/loinc_code2string.csv"
    rxnorm_tree_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/rxnorm/rxnorm_hierarchy.csv"
    rxnorm_map_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/rxnorm/rxnorm_code2string.csv"
    phecode_tree_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/icd_phecode/icd_phecode_hierarchy.csv"
    phecode_map_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/icd_phecode/icd_code2string.csv"
    cpt_tree_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/cpt_ccs/cpt_ccs_hierarchy.csv"
    cpt_map_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/cpt_ccs/cpt_code2string.csv"



    # umls_dataset = UMLSDataset(umls_folder="D:/Projects/CODER/deps/UMLS/2021AB/META",
    #                            model_name_or_path="monologg/biobert_v1.1_pubmed",
    #                            lang=None)
    # dataloader = fixed_length_dataloader(umls_dataset, num_workers=4)

    tree_dataset = TreeDataset(loinc_tree_path=loinc_tree_path, 
        loinc_map_path=loinc_map_path, 
        rxnorm_tree_path=rxnorm_tree_path,
        rxnorm_map_path=rxnorm_map_path,
        cpt_tree_path=cpt_tree_path,
        cpt_map_path=cpt_map_path,
        model_name_or_path="monologg/biobert_v1.1_pubmed")
    dataloader = fixed_length_dataloader(tree_dataset, num_workers=4)

    now_time = time()
    index = 0
    for batch in dataloader:
        print(index)
        index += 1
        print(time() - now_time)
        now_time = time()
        if index < 10:
            for item in batch:
                print(item.shape)
            #print(batch)
        else:
            import sys
            sys.exit()
