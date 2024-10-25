dataset="gsm8k"
inference_dir="./output/$dataset-inference"
reference_path="../lora-merging-datasets/test/$dataset.parquet"


PYTHONPATH="." python ./run/eval.py \
  --dataset $dataset \
  --inference_dir $inference_dir \
  --reference_path $reference_path
