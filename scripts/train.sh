set -e

dataset_name="mathqa"


model_name="Llama-3.2-1B-Instruct"
model_path="../models/$model_name"
output_dir="./output"

adapter_name=$dataset_name

local_dataset_dir="../lora-merging-datasets/"
training_partition="train"
disable_dataset_cache=true
dataset_num_proc=4

logging_steps=100
save_strategy="epoch"

max_seq_length=4096
disable_seq_length_filter=false

num_train_epochs=5
per_device_train_batch_size=2
gradient_accumulation_steps=2

use_peft_lora=true
lora_r=16
lora_alpha=64
lora_dropout=0.05
lora_target_modules="q_proj,v_proj"

use_flash_attn=false
gradient_checkpointing=false
use_reentrant=false
ddp_find_unused_parameters=false

fp16=false
bf16=true
use_4bit_quantization=true
bnb_4bit_compute_dtype="bfloat16"

report_to="wandb"
project_name="$model_name>>$dataset_name"
run_name="r${lora_r}.a${lora_alpha}"

CUDA_VISIBLE_DEVICES="2,3" \
PYTHONPATH="." \
accelerate launch --num_processes=2 --main_process_port=30500 ./run/main.py \
  --model_name_or_path $model_path \
  --adapter_name $adapter_name \
  --output_dir $output_dir \
  --local_dataset_dir $local_dataset_dir \
  --dataset_name $dataset_name \
  --disable_dataset_cache $disable_dataset_cache \
  --dataset_num_proc $dataset_num_proc \
  --disable_seq_length_filter $disable_seq_length_filter \
  --logging_steps $logging_steps \
  --save_strategy $save_strategy \
  --max_seq_length $max_seq_length \
  --num_train_epochs $num_train_epochs \
  --per_device_train_batch_size $per_device_train_batch_size \
  --gradient_accumulation_steps $gradient_accumulation_steps \
  --use_peft_lora $use_peft_lora \
  --lora_r $lora_r \
  --lora_alpha $lora_alpha \
  --lora_dropout $lora_dropout \
  --lora_target_modules $lora_target_modules \
  --use_flash_attn $use_flash_attn \
  --gradient_checkpointing $gradient_checkpointing \
  --use_reentrant $use_reentrant \
  --ddp_find_unused_parameters $ddp_find_unused_parameters \
  --fp16 $fp16 \
  --bf16 $bf16 \
  --use_4bit_quantization $use_4bit_quantization \
  --bnb_4bit_compute_dtype $bnb_4bit_compute_dtype \
  --report_to $report_to \
  --project_name $project_name \
  --run_name $run_name \
  --task train