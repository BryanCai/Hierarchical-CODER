import csv
from tqdm import tqdm
from random import shuffle
from collections import defaultdict

class LOINC(object):
    def __init__(self, tree_path, map_path):
        self.tree_path = tree_path
        self.map_path = map_path
        self.load()
        
    def load(self):
        self.parent = {}
        self.children = defaultdict(set)
        self.grandchildren = defaultdict(set)
        self.text = {}
        with open(self.tree_path) as f:
            reader = csv.reader(f)
            reader.__next__()
            for row in tqdm(reader, ascii=True):
                parent = row[2]
                current = row[3]
                if len(parent) > 0:
                    self.parent[current] = parent
                    self.children[parent].add(current)

        with open(self.map_path) as f:
            reader = csv.reader(f)
            reader.__next__()
            for row in tqdm(reader, ascii=True):
                current = row[0]
                text = row[9]
                self.text[current] = text

        for current in tqdm(list(self.children), ascii=True):
            for child in self.children[current]:
                self.grandchildren[current] = self.grandchildren[current].union(self.children[child])


    def __len__(self):
        return len(self.text)


class RXNORM(object):
    def __init__(self, rxnorm_path):
        self.rxnorm_path = rxnorm_path
        self.load()
        
    def load(self):
        self.parent = {}
        self.children = defaultdict(set)
        self.grandchildren = defaultdict(set)
        self.text = {}
        with open(self.rxnorm_path) as f:
            reader = csv.reader(f)
            reader.__next__()
            for row in tqdm(reader, ascii=True):
                rel = row[1]
                if rel in ['consists_of', 'has_precise_ingredient', 'has_part', 'has_ingredient']:
                    parent = row[0]
                    child = row[2]
                    parent_text = row[3]
                    child_text = row[4]
                if rel in ['form_of']:
                    child = row[0]
                    parent = row[2]
                    child_text = row[3]
                    parent_text = row[4]
                self.parent[child] = parent
                self.children[parent].add(child)
                self.text[parent] = parent_text
                self.text[child] = child_text

        for current in tqdm(list(self.children), ascii=True):
            for child in self.children[current]:
                self.grandchildren[current] = self.grandchildren[current].union(self.children[child])


    def __len__(self):
        return len(self.text)

class PHECODE(object):
    def __init__(self, phecode_path):
        self.phecode_path = phecode_path
        self.load()
        
    def load(self):
        self.parent = {}
        self.children = defaultdict(set)
        self.grandchildren = defaultdict(set)
        self.text = {}
        x = set()
        y = 0
        with open(self.phecode_path) as f:
            reader = csv.reader(f)
            reader.__next__()
            for row in tqdm(reader, ascii=True):
                phecode = row[2].lstrip('0')
                text = row[3]
                self.text[phecode] = text

        for phecode in self.text:
            if "." in phecode:
                parent = phecode[:-1]
                if parent[-1] == ".":
                    parent = parent[:-1]
                if parent in self.text:
                    self.parent[phecode] = parent
                    self.children[parent].add(phecode)
        
        for current in tqdm(list(self.children), ascii=True):
            for child in self.children[current]:
                self.grandchildren[current] = self.grandchildren[current].union(self.children[child])


    def __len__(self):
        return len(self.text)

class TREE(object):
    def __init__(self, tree_path, map_path):
        self.tree_path = tree_path
        self.map_path = map_path
        self.load()
        
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
            for row in tqdm(reader, ascii=True):
                current = row[0]
                text = row[1]
                self.text[current].add(text)

        for current in tqdm(list(self.children), ascii=True):
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
    phecode_tree_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/icd_phecode/icd_phecode_hierarchy.csv"
    phecode_map_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/icd_phecode/icd_code2string.csv"
    cpt_tree_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/cpt_ccs/cpt_ccs_hierarchy.csv"
    cpt_map_path = "D:/Projects/CODER/Hierarchical-CODER/data/codes/cpt_ccs/cpt_code2string.csv"

    loinc = TREE(loinc_tree_path, loinc_map_path)
    rxnorm = TREE(rxnorm_tree_path, rxnorm_map_path)
    phecode = TREE(phecode_tree_path, phecode_map_path)
    cpt = TREE(cpt_tree_path, cpt_map_path)
    for d in [loinc, rxnorm, phecode, cpt]:
        print(len(d.children))
        c = 0
        t = 0
        for i in d.children:
            if len(d.children[i]) > 0:
                t += 1
            c += len(d.children[i])
        print(c/t)
