import pandas as pd

class HierarchicalData():
    def __init__(self, icd9_phecode_path, icd10_phecode_path, loinc_path, rxnorm_path, cpt_ccs_path):
        self.icd9_phecode = pd.read_csv(icd9_phecode_path)
        self.icd10_phecode = pd.read_csv(icd10_phecode_path)
        self.loinc = pd.read_csv(loinc_path)
        self.rxnorm = pd.read_csv(rxnorm_path)
        self.cpt_ccs = pd.read_csv(cpt_ccs_path)
        self.load_icd_phecode()
        self.load_loinc()
        self.load_rxnorm()
        self.load_cpt_ccs()
    
    def load_icd_phecode(self):
        self.phecode2icd = dict()
        self.icd_string_list = []
        # icd9
        for idx in range(len(self.icd9_phecode)):
            phecode = self.icd9_phecode["PheCode"][idx]
            string = self.icd9_phecode["ICD9 String"][idx]
            if phecode not in self.phecode2icd:
                self.phecode2icd[phecode] = set()
            self.phecode2icd[phecode].update([string])
            self.icd_string_list.append(string)
        # icd10
        for idx in range(len(self.icd10_phecode)):
            phecode = self.icd10_phecode["PheCode"][idx]
            string = self.icd10_phecode["ICD10 String"][idx]
            if phecode not in self.phecode2icd:
                self.phecode2icd[phecode] = set()
            self.phecode2icd[phecode].update([string])
            self.icd_string_list.append(string)
        
        self.phecode_list = list(self.phecode2icd.keys())
        self.icd_string_list = list(set(self.icd_string_list))

        print('phecode number:', len(self.phecode_list))
        print('icd string number:', len(self.icd_string_list))
    
    def load_loinc(self):
        '''
        self.loinc_parent2child: all {parent: {child}} pairs in loinc
        self.loinc_part2term: all {loinc part: {loinc term}} pairs in loinc, terms are the leaf nodes
        self.loinc_code2string: all {code: string} pairs in loinc, including parts and terms
        e.g.:
        loinc part: LP14082-9 -> Bacteria | Bronchoalveolar lavage | Microbiology
        loinc term: 99932-6 -> Bacteria Fld Ql Auto
        '''
        self.loinc_parent2child = dict()
        self.loinc_part2term = dict()
        self.loinc_code2string = dict()
        for idx in range(len(self.loinc)):
            parent_code = self.loinc['IMMEDIATE_PARENT'][idx]
            loinc_code = self.loinc['CODE'][idx]
            loinc_string = self.loinc['CODE_TEXT'][idx]
            if parent_code not in self.loinc_parent2child:
                self.loinc_parent2child[parent_code] = set()
            self.loinc_parent2child[parent_code].update([loinc_code])
            if loinc_code[:2] != 'LP':
                if parent_code not in self.loinc_part2term:
                    self.loinc_part2term[parent_code] = set()
                self.loinc_part2term[parent_code].update([loinc_code])
            self.loinc_code2string[loinc_code] = loinc_string
        # grand_grandparents -> grandparents -> parents -> loinc terms
        self.loinc_parents = list(self.loinc_part2term.keys())


        print('number of loinc part + loinc term:', len(self.loinc_code2string))    

    def load_rxnorm(self):
        '''
        hierarchy: rxcui2 rela rxcui1   e.g. Sea-Omega has_precise_ingredient eicosapentaenoic acid
        {parent: child} = {rxcui1: {rxcui2}} -> two drugs containing same ingredient are similar
        '''
        self.rxnorm_parent2child = dict()
        for idx in range(len(self.rxnorm)):
            parent_string = self.rxnorm["RXCUI1_STR"][idx]
            child_string = self.rxnorm["RXCUI2_STR"][idx]
            if parent_string not in self.rxnorm_parent2child:
                self.rxnorm_parent2child[parent_string] = set()
            self.rxnorm_parent2child[parent_string].update([child_string])
        self.rxnorm_parents = list(self.rxnorm_parent2child.keys())

        print('number of rxnorm parents:', len(self.rxnorm_parents))

    def load_cpt_ccs(self):
        self.ccs2cpt_code = dict()
        for idx in range(len(self.cpt_ccs)):
            cpt_range = self.cpt_ccs['Code Range'][idx][1:-1].split('-')
            ccs = self.cpt_ccs['CCS'][idx]
            if ccs not in self.ccs2cpt_code:
                self.ccs2cpt_code[ccs] = set()
            if cpt_range[0] == cpt_range[1]:
                self.ccs2cpt_code[ccs].update([cpt_range[0]])
            else:
                cpt_range = list(range(int(cpt_range[0]), int(cpt_range[1])+1))
                self.ccs2cpt_code[ccs].update([str(cpt) for cpt in cpt_range])
        
        print('ccs number:', len(self.ccs2cpt_code))

            

if __name__ == '__main__':
    data = HierarchicalData(
        icd9_phecode_path='icd_phecode/phecode_icd9_rolled.csv',
        icd10_phecode_path='icd_phecode/phecode_icd10.csv',
        loinc_path='loinc/AccessoryFiles/MultiAxialHierarchy/MultiAxialHierarchy.csv',
        rxnorm_path='rxnorm/rxnorm_hierarchy_w_cuis_2019umls.csv',
        cpt_ccs_path='cpt_ccs/CCS_services_procedures_v2021-1.csv'
    )
