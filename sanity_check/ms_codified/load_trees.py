import csv
from tqdm import tqdm
from random import shuffle
from collections import defaultdict
import ahocorasick
import pandas as pd
import re

class TREE(object):
    def __init__(self, tree_path, map_path, eval_data_path):
        self.tree_path = tree_path
        self.map_path = map_path
        self.eval_data = pd.read_csv(eval_data_path)
        self.eval_terms = self.all_eval_terms()
        self.load()
    
    def all_eval_terms(self):
        terms = list(self.eval_data['STR1']) + list(self.eval_data['STR2']) + list(self.eval_data['neg str1']) + list(self.eval_data['neg str2'])
        terms = list(set(terms))
        terms = [str(term) for term in terms if len(str(term))>0]
        terms += [self.clean(term) for term in terms]
        terms += [term.lower() for term in terms]
        terms = list(set(terms))
        print("Number of terms in eval data:", len(terms))
        trems_trie = ahocorasick.Automaton()
        for term in terms:
            trems_trie.add_word(term, '')
        return trems_trie
    
    def clean(self, term, lower=True, clean_NOS=True, clean_bracket=True, clean_dash=True):
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

    def load(self):
        self.parent = {}
        self.children = defaultdict(set)
        self.grandchildren = defaultdict(set)
        self.text = defaultdict(set)
        with open(self.tree_path) as f:
            reader = csv.reader(f)
            reader.__next__()
            for row in tqdm(reader, ascii=True):
                parent = row[0]
                current = row[1]
                if len(parent) > 0 and len(current) > 0:
                    self.parent[current] = parent
                    self.children[parent].add(current)

        with open(self.map_path) as f:
            reader = csv.reader(f)
            reader.__next__()
            for row in reader:
                current = row[0]
                text = row[1]
                if text in self.eval_terms or text.lower() in self.eval_terms or self.clean(text) in self.eval_terms:
                    continue
                self.text[current].add(self.clean(text))

        for current in list(self.children):
            for child in self.children[current]:
                self.grandchildren[current] = self.grandchildren[current].union(self.children[child])

        self.text = self.clean_set_dict(self.text)

    def clean_set_dict(self, d):
        out = {}
        for i in d:
            if len(d[i]) > 0:
                out[i] = tuple(d[i])
        return out


    def __len__(self):
        return len(self.text)


if __name__ == "__main__":

    loinc_tree_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/loinc/loinc_hierarchy.csv"
    loinc_map_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/loinc/loinc_code2string.csv"
    rxnorm_tree_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/rxnorm/rxnorm_hierarchy.csv"
    rxnorm_map_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/rxnorm/rxnorm_code2string.csv"
    phecode_tree_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/icd_phecode/phecode_hierarchy.csv"
    phecode_map_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/icd_phecode/phecode2icd_string.csv"
    cpt_tree_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/cpt_ccs/cpt_ccs_hierarchy.csv"
    cpt_map_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/cpt_ccs/cpt_code2string.csv"

    loinc = TREE(loinc_tree_path, loinc_map_path)
    rxnorm = TREE(rxnorm_tree_path, rxnorm_map_path)
    phecode = TREE(phecode_tree_path, phecode_map_path)
    cpt = TREE(cpt_tree_path, cpt_map_path)
    for d in [loinc, rxnorm, phecode, cpt]:
        print(d.tree_path)
        print(len(set(d.parent.keys()).union(set(d.children.keys()))))
