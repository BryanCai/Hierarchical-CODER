### LOINC

Laboratory observations. Hierarchy is in the file `loinc/AccessoryFiles/MultiAxialHierarchy/MultiAxialHierarchy.csv`. It defines the hierarchy as a tree. Each row defines one node of the tree, including path to root, parent, code, and string.

All mappings between codes and strings are in `loinc/LoincTableCore/LoincTableCore.csv`.

### RxNorm

Drugs. Hierarchy is in the file `rxnorm/rxnorm_hierarchy_w_cuis_2019umls.csv`

### Phecode

Diseases. Phecode is hierarchical. Mappings between ICD strings and phecodes is in `icd_phecode/phecode_icd9_rolled.csv` and `icd_phecode/phecode_icd10.csv`.

### CPT-CCS

Procedures. CPT codes are rolled up to CCS codes, which is listed in `cpt_ccs/CCS_services_procedures_v2021-1.csv`. Haven't found the mapping between CPT codes and strings.
<!-- ### ICD-10-PCS

Procedures. The codes and corresponding strings are listed in `icd10_pcs/icd10pcs_codes_2022.txt`. ICD-10-PCS with CCS categories is in `ccs_pr_icd10pcs_2020_1.csv`. (Are there other hierarchies for procedure codes? CCS category has only one level.) -->