import json
import sys


def run(file):
    with open(file, 'r') as f:
        data = json.load(f)


    print(file)
    for i in [
              "Similar.csv",
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
              'ALL_CAUSITIVE',
              'ALL_MAY_CAUSE_OR_TREAT',
              'method_of',
              'classifies',
              'DDX'
              ]:
        print(data[i])

    print("\n")



    for i in [
              'component_of',
              'classified_as',
              'finding_site_of',
              'translation_of',
              'isa',
              'subset_includes_concept',
              'inverse_isa',
              'has_ingredient',
              'concept_in_subset',
              'mapped_from',
              'has_member',
              'member_of',
              'has_translation',
              'expanded_form_of',
              'mapped_to',
              'has_inactive_ingredient',
              'ingredient_of',
              'inactive_ingredient_of',
              'has_finding_site',
              'classifies',
              'active_moiety_of',
              'has_component',
              'has_active_ingredient',
              'active_ingredient_of',
              'has_active_moiety',
              'has_class',
              'has_expanded_form',
              'class_of'
              ]:
        print(data[i])

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

    for i in [
              "cadec",
              ]:
        print(data[i][0])
        print(data[i][1])
        print(data[i][2])
        print(data[i][3])

    print("\n")



if __name__ == '__main__':

    output_path_list = [
                        "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/triplet_umls_full/output.json",
                        "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/triplet_umls_rela_tree_full/output.json",
                        "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/clogit_01_triplet_umls_rela_tree_full_lr_6/output.json",
                        "/home/tc24/BryanWork/saved_models/sapbert_hierarchical/clogit_01_triplet_umls_rela_tree_100_lr_6/output.json",
                        # "/home/tc24/BryanWork/saved_models/output_coder_base/output_300000_0.json",
                        # "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/sapbert_0.json",
                        # "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/coder_0.json",
                        # "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/biobert1_1_0.json",
                        # "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/biogpt_0.json",
                        # "/home/tc24/BryanWork/CODER/unified_eval/fixed_model_eval/distilbert_0.json",
                        ]

    for i in output_path_list:
        run(i)
