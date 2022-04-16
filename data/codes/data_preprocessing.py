import pandas as pd

class HierarchicalData():
    def __init__(self, icd9_phecode_path, icd10_phecode_path, loinc_path, rxnorm_path, cpt_ccs_path, cpt_path):
        self.icd9_phecode = pd.read_csv(icd9_phecode_path, dtype='str').dropna().reset_index(drop=True)
        self.icd10_phecode = pd.read_csv(icd10_phecode_path, dtype='str').dropna().reset_index(drop=True)
        self.loinc = pd.read_csv(loinc_path)
        self.rxnorm = pd.read_csv(rxnorm_path)
        self.cpt_ccs = pd.read_csv(cpt_ccs_path)
        self.cpt = pd.read_csv(cpt_path, sep='\t')
        self.cpt2string = {str(key): value for key, value in zip(self.cpt['CODE'], self.cpt['STR'])}
        self.load_icd_phecode()
        self.load_loinc()
        self.load_rxnorm()
        self.load_cpt_ccs()
    
    def load_icd_phecode(self):
        self.phecode2icd = dict()
        self.icd2string = dict()
        self.icd_string_list = []
        # icd9
        for idx in range(len(self.icd9_phecode)):
            phecode = self.icd9_phecode["PheCode"][idx]
            icd9 = self.icd9_phecode["ICD9"][idx]
            string = self.icd9_phecode["ICD9 String"][idx]
            if phecode not in self.phecode2icd:
                self.phecode2icd[phecode] = set()
            self.phecode2icd[phecode].update([icd9])
            self.icd2string[icd9] = string
            self.icd_string_list.append(string)
        # icd10
        for idx in range(len(self.icd10_phecode)):
            phecode = self.icd10_phecode["PheCode"][idx]
            icd10 = self.icd10_phecode["ICD10"][idx]
            string = self.icd10_phecode["ICD10 String"][idx]
            if phecode not in self.phecode2icd:
                self.phecode2icd[phecode] = set()
            self.phecode2icd[phecode].update([icd10])
            self.icd2string[icd10] = string
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
        self.rxnorm_code2string = dict()
        for idx in range(len(self.rxnorm)):
            parent_code = self.rxnorm['RXCUI1'][idx]
            child_code = self.rxnorm['RXCUI2'][idx]
            parent_string = self.rxnorm["RXCUI1_STR"][idx]
            child_string = self.rxnorm["RXCUI2_STR"][idx]
            if parent_code not in self.rxnorm_parent2child:
                self.rxnorm_parent2child[parent_code] = set()
            self.rxnorm_parent2child[parent_code].update([child_code])
            self.rxnorm_code2string[parent_code] = parent_string
            self.rxnorm_code2string[child_code] = child_string
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
                if cpt_range[0] in self.cpt2string:
                    self.ccs2cpt_code[ccs].update([cpt_range[0]])
            else:
                cpt_range = list(range(int(cpt_range[0]), int(cpt_range[1])+1))
                for cpt in cpt_range:
                    if str(cpt) in self.cpt2string:
                        self.ccs2cpt_code[ccs].update([str(cpt)])
        
        print('ccs number:', len(self.ccs2cpt_code))
        # print(self.ccs2cpt_code)

    def print_to_file(self):
        # icd-phecode
        phecode_parent2child_data = []
        for phe1 in self.phecode_list:
            for phe2 in self.phecode_list:
                try:
                    if len(phe2) > len(phe1):
                        if len(phe1)==3 and phe2[:3]==phe1[:3]:
                            phecode_parent2child_data.append([phe1, phe2])
                        if len(phe1)==5 and phe2[:5]==phe1[:5]:
                            phecode_parent2child_data.append([phe1, phe2])
                except:
                    print(phe1, phe2)
        df1 = pd.DataFrame(phecode_parent2child_data, columns=['Parent', 'Child'], dtype='str')
        df1 = df1.dropna()
        df1.to_csv('phecode_hierarchy.csv', index=False)


        phecode2icdstring_data = []
        for phecode, icd_set in self.phecode2icd.items():
            for icd in icd_set:
                phecode2icdstring_data.append([phecode, self.icd2string[icd]])
        df2 = pd.DataFrame(phecode2icdstring_data, columns=['Phecode', 'ICD string'], dtype='str')
        df2 = df2.dropna()
        # print(df.head)
        df2.to_csv('phecode2icd_string.csv', index=False)
        

        # loinc
        loinc_data = []
        for parent, children in self.loinc_parent2child.items():
            for child in children:
                loinc_data.append([parent, child])
        df1 = pd.DataFrame(loinc_data, columns=['Parent', 'Child'], dtype='str')
        df1 = df1.dropna()
        df1.to_csv('loinc_hierarchy.csv', index=False)
        
        loinc_string_data = [[code, string] for code, string in self.loinc_code2string.items()]
        df2 = pd.DataFrame(loinc_string_data, columns=['Loinc code', 'String'], dtype='str')
        df2 = df2.dropna()
        df2.to_csv('loinc_code2string.csv', index=False)

        # rxnorm
        rxnorm_data = []
        for parent, children in self.rxnorm_parent2child.items():
            for child in children:
                rxnorm_data.append([parent, child])
        df1 = pd.DataFrame(rxnorm_data, columns=['Parent', 'Child'], dtype='str')
        df1 = df1.dropna()
        df1.to_csv('rxnorm_hierarchy.csv', index=False)
        
        rxnorm_string_data = [[code, string] for code, string in self.rxnorm_code2string.items()]
        df2 = pd.DataFrame(rxnorm_string_data, columns=['Rxnorm code', 'String'], dtype='str')
        df2 = df2.dropna()
        df2.to_csv('rxnorm_code2string.csv', index=False)      

        # cpt-ccs
        cpt_ccs_data = []
        for ccs, cpt_set in self.ccs2cpt_code.items():
            for cpt in cpt_set:
                cpt_ccs_data.append([ccs, cpt])
        df1 = pd.DataFrame(cpt_ccs_data, columns=['CCS', 'CPT'], dtype='str')
        df1 = df1.dropna()
        df1.to_csv('cpt_ccs_hierarchy.csv', index=False)

        cpt_string_data = [[code, string] for code, string in self.cpt2string.items()]
        df2 = pd.DataFrame(cpt_string_data, columns=['CPT code', 'String'], dtype='str')
        df2 = df2.dropna()
        df2.to_csv('cpt_code2string.csv', index=False)      

        




if __name__ == '__main__':
    data = HierarchicalData(
        icd9_phecode_path='icd_phecode/phecode_icd9_rolled.csv',
        icd10_phecode_path='icd_phecode/phecode_icd10.csv',
        loinc_path='loinc/AccessoryFiles/MultiAxialHierarchy/MultiAxialHierarchy.csv',
        rxnorm_path='rxnorm/rxnorm_hierarchy_w_cuis_2019umls.csv',
        cpt_ccs_path='cpt_ccs/CCS_services_procedures_v2021-1.csv',
        cpt_path='cpt_ccs/CUI_CPT.tsv'
    )
    data.print_to_file()
