python train.py --model_name_or_path monologg/biobert_v1.1_pubmed --tree_dir ../../data/cleaned/all --eval_data_path ../../eval/all_relations_with_neg.csv --coder_path GanjinZero/coder_eng --save_step 5000 --tree_batch_size 8 --max_steps 30000 --output_dir output_balance --warmup_steps 0