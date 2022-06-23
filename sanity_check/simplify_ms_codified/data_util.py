import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from load_trees import TREE
from torch.utils.data import Dataset, DataLoader
import random
from torch.utils.data.sampler import RandomSampler
from time import time
import json
from pathlib import Path
from sampler_util import FixedLengthBatchSampler, my_collate_fn
from random import sample

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


class TreeDataset(Dataset):
    def __init__(self, tree_dir, model_name_or_path, max_neg_samples=8, max_length=32, eval_data_path=None):
        tree_dir = Path(tree_dir)
        tree_subdirs = [f for f in tree_dir.iterdir() if f.is_dir()]
        self.trees = {}
        for tree_subdir in tree_subdirs:
            print(tree_subdir.name)
            self.trees[tree_subdir.name] = TREE(tree_subdir/"hierarchy.csv", tree_subdir/"code2string.csv", eval_data_path)
        self.obj_list = []
        self.len = 0
        for tree in self.trees:
            if tree == 'phecode':
                self.obj_list += [(i, tree) for i in self.trees[tree].text.keys()] * 100
                self.len += len(self.trees[tree]) * 100
            elif tree == 'cpt':
                self.obj_list += [(i, tree) for i in self.trees[tree].text.keys()] * 15
                self.len += len(self.trees[tree]) * 15
            else:


            #     print('before', len(self.obj_list))
            #     self.obj_list += [(i, tree) for i in self.trees[tree].text if '.' in i and len(i.split(".")[1]) == 2]
            #     print('after', len(self.obj_list))
            #     self.len += len([(i, tree) for i in self.trees[tree].text if '.' in i and len(i.split(".")[1]) == 2])
            # if tree != 'phecode':
            #     continue
                self.obj_list += [(i, tree) for i in self.trees[tree].text.keys()]
                self.len += len(self.trees[tree])
        temp = []
        for tree in self.trees:
            temp += [(i, tree) for i in self.trees[tree].text.keys()]
        self.anchorid2cont_idx = {item[0]:idx for idx, item in enumerate(temp)}
        self.max_neg_samples = max_neg_samples
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def __getitem__(self, index):
        anchor_id, tree = self.obj_list[index]
        if anchor_id not in self.trees[tree].text:
            return [], [], []
        neg_samples = []

        if tree == 'phecode':
            if "." not in anchor_id:
                level = 3
            else:
                level = 3 - len(anchor_id.split(".")[1])
            neg_samples += [(i, level-1) for i in self.trees[tree].children[anchor_id]]
            if anchor_id in self.trees[tree].parent:
                parent = self.trees[tree].parent[anchor_id]
                neg_samples += [(parent, level)]
                neg_samples += [(i, level) for i in self.trees[tree].children[parent] if i != anchor_id]
            pos_samples = [(anchor_id, level-1)] * self.max_neg_samples

        else:    
            neg_samples += [(i, 1) for i in self.trees[tree].children[anchor_id]]
            if anchor_id in self.trees[tree].parent:
                parent = self.trees[tree].parent[anchor_id]
                neg_samples += [(parent, 2)]
                neg_samples += [(i, 1) for i in self.trees[tree].children[parent] if i != anchor_id]
            pos_samples = [(anchor_id, 0)] * self.max_neg_samples

        neg_samples = [i for i in neg_samples if i[0] in self.trees[tree].text]

        if len(neg_samples) > self.max_neg_samples:
            neg_samples = random.sample(neg_samples, self.max_neg_samples)
        if len(neg_samples) < self.max_neg_samples:
            neg_samples += [(i, 3) for i in sample(self.trees[tree].text.keys(), self.max_neg_samples-len(neg_samples))]
        
        far_neg_samples = [(i, 3) for i in sample(self.trees[tree].text.keys(), self.max_neg_samples)]

        all_samples = pos_samples + neg_samples + far_neg_samples


        if len(neg_samples) == 0:
            return [], [], []

        all_samples_id, all_samples_dist = list(zip(*all_samples))
        # neg_samples_id, neg_samples_dist = list(zip(*neg_samples))

        anchor_string = random.choice(self.trees[tree].text[anchor_id])
        # neg_samples_string = [random.choice(self.trees[tree].text[i]) for i in neg_samples_id]
        all_samples_string = [random.choice(self.trees[tree].text[i]) for i in all_samples_id]
        # if tree == 'phecode':
        #     all_samples_id = [i.split('.')[0] if '.' in i and i.split('.')[0] in self.anchorid2cont_idx else i for i in all_samples_id]


        anchor_input_id = self.tokenize_one(anchor_string)
        # neg_samples_input_id = [self.tokenize_one(s) for s in neg_samples_string]
        all_samples_input_id = [self.tokenize_one(s) for s in all_samples_string]

        all_samples_cont_idx = [self.anchorid2cont_idx[i] for i in all_samples_id]

        # return [anchor_input_id]*len(neg_samples_input_id), neg_samples_input_id, neg_samples_dist
        return [anchor_input_id]*len(all_samples_input_id), all_samples_input_id, all_samples_dist

    def __len__(self):
        return self.len


    def tokenize_one(self, string):
        return self.tokenizer.encode_plus(string, max_length=self.max_length, truncation=True, padding='max_length')['input_ids']


def fixed_length_dataloader(dataset, fixed_length=96, num_workers=0):
    base_sampler = RandomSampler(dataset)
    batch_sampler = FixedLengthBatchSampler(
        sampler=base_sampler, fixed_length=fixed_length, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler,
                            collate_fn=my_collate_fn, num_workers=num_workers, pin_memory=True)
    return dataloader
