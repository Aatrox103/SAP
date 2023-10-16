# 0
python finetune.py \
    --base_model='decapoda-research/llama-13b-hf' \
    --num_epochs=20 \
    --cutoff_len=512 \
    --group_by_length \
    --data_path 'finetune_data/30/alpaca_safety.json' \
    --output_dir './lora-alpaca/30/alpaca_13B_finetune_without_regen' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8 \
    --val_set_size=0 \
    --resume_from_checkpoint='lora-alpaca/alpaca_13B'

# 1
python finetune.py \
    --base_model='decapoda-research/llama-13b-hf' \
    --num_epochs=20 \
    --cutoff_len=512 \
    --group_by_length \
    --data_path 'finetune_data/30/alpaca_safety.json' \
    --output_dir './lora-alpaca/30/alpaca_13B_finetune1_without_regen' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8 \
    --val_set_size=0 \
    --resume_from_checkpoint='lora-alpaca/30/alpaca_13B_finetune_without_regen'

# 2
python finetune.py \
    --base_model='decapoda-research/llama-13b-hf' \
    --num_epochs=20 \
    --cutoff_len=512 \
    --group_by_length \
    --data_path 'finetune_data/30/alpaca_safety.json' \
    --output_dir './lora-alpaca/30/alpaca_13B_finetune2_without_regen' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8 \
    --val_set_size=0 \
    --resume_from_checkpoint='lora-alpaca/30/alpaca_13B_finetune1_without_regen'

# 3
python finetune.py \
    --base_model='decapoda-research/llama-13b-hf' \
    --num_epochs=20 \
    --cutoff_len=512 \
    --group_by_length \
    --data_path 'finetune_data/30/alpaca_safety.json' \
    --output_dir './lora-alpaca/30/alpaca_13B_finetune3_without_regen' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=8 \
    --val_set_size=0 \
    --resume_from_checkpoint='lora-alpaca/30/alpaca_13B_finetune2_without_regen'