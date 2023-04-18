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


    for i in [
              "('CUI-CUI', 'procedure_diagnoses')",
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

    print("\n")


    for i in [
              "(0, 1)",
              "(0, 2)",
              "(0, 3)",
              "(1, 2)",
              "(1, 3)",
              "(2, 3)",
              ]:
        print(data[i])

    print("\n")

    for i in [
              "cadec",
              ]:
        print(data[i][0])
        print(data[i][1])

    print("\n")

    for i in [
              "example0_cos_sim",
              "example1_cos_sim",
              "example2_cos_sim",
              "example3_cos_sim",
              "example4_cos_sim",
              ]:
        print(data[i])

    print("\n")

if __name__ == '__main__':
    output_path_list = [
                        "/home/tc24/BryanWork/saved_models/output_coder_base/output2_300000.json",
#                        "/home/tc24/BryanWork/saved_models/output_unified_ms/output2_300000.json",
                        "/home/tc24/BryanWork/saved_models/old/output_unified_3/output2_300000.json",
                        "/home/tc24/BryanWork/saved_models/old/output_unified_ft_5/output2_20000.json",
#                        "/home/tc24/BryanWork/saved_models/output_unified_ft_7/output2_10000.json",
                        "/home/tc24/BryanWork/saved_models/output_unified_ft_8/output2_10000.json",
#                        "/home/tc24/BryanWork/saved_models/output_unified_ft_9/output2_10000.json",
                        "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/sapbert.json",
                        "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/coder.json",
                        "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/biobert1_1.json",
                        "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/biogpt.json",
                        "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/distilbert.json",
                        ]

    for i in output_path_list:
        run(i)
