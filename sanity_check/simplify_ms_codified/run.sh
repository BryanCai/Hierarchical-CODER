python train.py --model_name_or_path monologg/biobert_v1.1_pubmed --tree_dir ../../data/cleaned/all --eval_data_path ../../eval/all_relations_with_neg.csv --coder_path /media/sdb1/Zengsihang/Hier_CODER/Hierarchical_CODER_new/output_ori_coder_filter/model_400000_bert.pth --save_step 5000 --tree_batch_size 16 --max_steps 30000 --output_dir output_balance_modified_0.9024 --warmup_steps 0
# python train.py --model_name_or_path GanjinZero/coder_eng --tree_dir ../data/cleaned/train --eval_data_path ../eval/all_relations_with_neg.csv --coder_path GanjinZero/coder_eng --save_step 5000 --tree_batch_size 16 --max_steps 100000 --output_dir output_onlinecoder