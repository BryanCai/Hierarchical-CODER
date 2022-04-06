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


if __name__ == "__main__":
    loinc_tree_path = "D:/Projects/CODER/deps/codes/loinc/AccessoryFiles/MultiAxialHierarchy/MultiAxialHierarchy.csv"
    loinc_map_path = "D:/Projects/CODER/deps/codes/loinc/LoincTableCore/LoincTableCore.csv"
    rxnorm_path = "D:/Projects/CODER/deps/codes/rxnorm/rxnorm_hierarchy_w_cuis_2019umls.csv"
    phecode_path = "D:/Projects/CODER/deps/codes/icd_phecode/phecode_icd9_rolled.csv"
    x = PHECODE(phecode_path)

