set -e

dataset_name="math"


model_name="Llama-3.2-1B-Instruct"
model_path="../models/$model_name"
output_dir="./output"
inference_dir="$output_dir/$dataset_name-inference"

adapter_name=$dataset_name

local_dataset_dir="../lora-merging-datasets/"
disable_dataset_cache=true
dataset_num_proc=4

max_seq_length=4096
max_new_tokens=1024
disable_seq_length_filter=false

fp16=false
bf16=true
use_4bit_quantization=true
bnb_4bit_compute_dtype="bfloat16"

report_to="none"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="." \
python ./run/main.py \
  --model_name_or_path $model_path \
  --adapter_name $adapter_name \
  --output_dir $output_dir \
  --inference_dir $inference_dir \
  --local_dataset_dir $local_dataset_dir \
  --dataset_name $dataset_name \
  --disable_dataset_cache $disable_dataset_cache \
  --dataset_num_proc $dataset_num_proc \
  --disable_seq_length_filter $disable_seq_length_filter \
  --max_seq_length $max_seq_length \
  --max_new_tokens $max_new_tokens \
  --fp16 $fp16 \
  --bf16 $bf16 \
  --use_4bit_quantization $use_4bit_quantization \
  --bnb_4bit_compute_dtype $bnb_4bit_compute_dtype \
  --report_to $report_to \
  --task infer-vllm