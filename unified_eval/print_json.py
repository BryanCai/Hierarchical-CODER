import json
import sys


def run(file):
    with open(file, 'r') as f:
        data = json.load(f)


    print(file)
    for i in ["Similar.csv",
              "Relate.csv",
              ]:
        print(data[i])

    print("\n")

    for i in ["('PheCode-PheCode', 'similarity')",
              "('PheCode-PheCode', 'related')",
              "('PheCode-RXNORM', 'related')",
              "('RXNORM-RXNORM', 'similarity')",
              "('RXNORM-RXNORM', 'related')",
              "('LOINC-LOINC', 'similarity')",
              "('PheCode-CCS', 'related')",
              "('PheCode-LOINC', 'related')",

              ]:
        print(data[i])
    print("\n")


    for i in ["('CUI-CUI', 'procedure_diagnoses')",
              "('CUI-CUI', 'lab_diagnoses')",
              "('CUI-CUI', 'belong(s) to the category of')",
              "('CUI-CUI', 'is a subtype of')",
              "('CUI-CUI', 'may cause')",
              "('CUI-CUI', 'see also')",
              "('CUI-CUI', 'may be allelic with')",
              "('CUI-CUI', 'is associated with')",
              "('CUI-CUI', 'is a risk factor for')",
              "('CUI-CUI', 'may contraindicate')",
              "('CUI-CUI', 'interacts with')",
              "('CUI-CUI', 'belongs to the drug family of')",
              "('CUI-CUI', 'is a category subset of')",
              "('CUI-CUI', 'belongs to drug super-family')",
              "('CUI-CUI', 'is a vector for')",
              "('CUI-CUI', 'is an ingredient of')",
              "('CUI-CUI', 'ddx')",
              "('CUI-CUI', 'may be caused by')"
              ]:
        print(data[i])


if __name__ == '__main__':
    # file = sys.argv[1]
    output_path_list = ["sapbert.json",
                        "coder.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_1/output_20000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_1/output_360000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_3/outputl_20000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_3/output_300000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_ft_1/output_1000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_ft_1/output_7000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_ft_2/output_1000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_ft_2/output_10000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_ft_3/output_2000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_ft_3/output_10000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_ft_4/output_2000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_ft_4/output_10000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_ft_5/output_10000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_ft_5/output_20000.json",
                        ]

    for i in output_path_list:
        run(i)