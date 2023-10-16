python finetune.py \
    --base_model='decapoda-research/llama-13b-hf' \
    --num_epochs=20 \
    --cutoff_len=512 \
    --group_by_length \
    --data_path 'finetune_data/30/alpaca_safety.json' \
    --output_dir './lora-alpaca/30/alpaca_13B_finetune' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8 \
    --val_set_size=0 \
    --resume_from_checkpoint='lora-alpaca/alpaca_13B'


python alpaca_generate.py --data_path 30/GPT_30_fraud --output_file_name alpaca_13B_finetune --lora_weights lora-alpaca/30/alpaca_13B_finetune --base_model decapoda-research/llama-7b-hf
python alpaca_generate.py --data_path 30/GPT_30_politics --output_file_name alpaca_13B_finetune --lora_weights lora-alpaca/30/alpaca_13B_finetune --base_model decapoda-research/llama-7b-hf
python alpaca_generate.py --data_path 30/GPT_30_pornography_sexual_minors --output_file_name alpaca_13B_finetune --lora_weights lora-alpaca/30/alpaca_13B_finetune --base_model decapoda-research/llama-7b-hf
python alpaca_generate.py --data_path 30/GPT_30_race --output_file_name alpaca_13B_finetune --lora_weights lora-alpaca/30/alpaca_13B_finetune --base_model decapoda-research/llama-7b-hf
python alpaca_generate.py --data_path 30/GPT_30_religion --output_file_name alpaca_13B_finetune --lora_weights lora-alpaca/30/alpaca_13B_finetune --base_model decapoda-research/llama-7b-hf
python alpaca_generate.py --data_path 30/GPT_30_suicide --output_file_name alpaca_13B_finetune --lora_weights lora-alpaca/30/alpaca_13B_finetune --base_model decapoda-research/llama-7b-hf
python alpaca_generate.py --data_path 30/GPT_30_terrorism --output_file_name alpaca_13B_finetune --lora_weights lora-alpaca/30/alpaca_13B_finetune --base_model decapoda-research/llama-7b-hf
python alpaca_generate.py --data_path 30/GPT_30_violence --output_file_name alpaca_13B_finetune --lora_weights lora-alpaca/30/alpaca_13B_finetune --base_model decapoda-research/llama-7b-hf


python gpt_eval.py --data_path 30/GPT_30_fraud --eval_filename_prefix alpaca_13B_finetune
python gpt_eval.py --data_path 30/GPT_30_politics --eval_filename_prefix alpaca_13B_finetune
python gpt_eval.py --data_path 30/GPT_30_pornography_sexual_minors --eval_filename_prefix alpaca_13B_finetune
python gpt_eval.py --data_path 30/GPT_30_race --eval_filename_prefix alpaca_13B_finetune
python gpt_eval.py --data_path 30/GPT_30_religion --eval_filename_prefix alpaca_13B_finetune
python gpt_eval.py --data_path 30/GPT_30_suicide --eval_filename_prefix alpaca_13B_finetune
python gpt_eval.py --data_path 30/GPT_30_terrorism --eval_filename_prefix alpaca_13B_finetune
python gpt_eval.py --data_path 30/GPT_30_violence --eval_filename_prefix alpaca_13B_finetune

python regenerate_case.py --selected_topic fraud --score_filename_prefix alpaca_13B_finetune --expected_cases 30
python regenerate_case.py --selected_topic politics  --score_filename_prefix alpaca_13B_finetune --expected_cases 30
python regenerate_case.py --selected_topic pornography_sexual_minors  --score_filename_prefix alpaca_13B_finetune --expected_cases 30
python regenerate_case.py --selected_topic race  --score_filename_prefix alpaca_13B_finetune --expected_cases 30
python regenerate_case.py --selected_topic religion  --score_filename_prefix alpaca_13B_finetune --expected_cases 30
python regenerate_case.py --selected_topic suicide  --score_filename_prefix alpaca_13B_finetune --expected_cases 30
python regenerate_case.py --selected_topic terrorism  --score_filename_prefix alpaca_13B_finetune --expected_cases 30
python regenerate_case.py --selected_topic violence  --score_filename_prefix alpaca_13B_finetune --expected_cases 30
python generate_finetune_data.py --filename_prefix alpaca_13B_finetune --expected_cases 30 --init_bool 0