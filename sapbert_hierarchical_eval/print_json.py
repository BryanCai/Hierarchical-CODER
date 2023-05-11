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
              "classified_as",
              "translation_of",
              "isa",
              "inverse_isa",
              "mapped_from",
              "has_member",
              "member_of",
              "has_translation",
              "expanded_form_of",
              "mapped_to",
              "has_inactive_ingredient",
              "inactive_ingredient_of",
              "classifies",
              "has_expanded_form"
              ]:
        print(data[i])

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
                        "/home/tc24/BryanWork/saved_models/output_unified_pubmed_base/output2_200000.pth",
                        "/home/tc24/BryanWork/saved_models/output_unified_ft_11/output2_10000.pth",
                        "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/sapbert.json",
                        "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/coder.json",
                        "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/biobert1_1.json",
                        "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/biogpt.json",
                        "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/distilbert.json",
                        ]

    for i in output_path_list:
        run(i)
