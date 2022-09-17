import json
import sys


def run(file):
    with open(file, 'r') as f:
        data = json.load(f)
    if "similar1" in data:
        for i in ["similar1", "similar2", "related1", "related2"]:
            print(data[i])


    if "('similarity', 'PheCode-PheCode')" in data:
        for i in ["('similarity', 'PheCode-PheCode')",
                  "('related', 'PheCode-PheCode')",
                  "('related', 'PheCode-RXNORM')",
                  "('related', 'RXNORM-RXNORM')",
                  "('similarity', 'LAB-LAB')",
                  "('related', 'PheCode-CCS')",
                  "('related', 'PheCode-LAB')",
                  "('related', 'CUI-CUI', 'belong(s) to the category of')",
                  "('related', 'CUI-CUI', 'is a subtype of')",
                  "('related', 'CUI-CUI', 'belongs to the drug family of')",
                  "('related', 'CUI-CUI', 'is a category subset of')",
                  "('related', 'CUI-CUI', 'belongs to drug super-family')",
                  "('related', 'CUI-CUI', 'is a vector for')",
                  "('related', 'CUI-CUI', 'is an ingredient of')",
                  "('related', 'CUI-CUI', 'procedure_diagnoses')",
                  "('related', 'CUI-CUI', 'lab_diagnoses')",
                  "('related', 'CUI-CUI', 'may cause')",
                  "('related', 'CUI-CUI', 'see also')",
                  "('related', 'CUI-CUI', 'may be allelic with')",
                  "('related', 'CUI-CUI', 'is associated with')",
                  "('related', 'CUI-CUI', 'is a risk factor for')",
                  "('related', 'CUI-CUI', 'may contraindicate')",
                  "('related', 'CUI-CUI', 'interacts with')",
                  "('related', 'CUI-CUI', 'ddx')",
                  "('related', 'CUI-CUI', 'may be caused by')"
                ]:
            print(data[i])



if __name__ == '__main__':
    file = sys.argv[1]
    run(file)