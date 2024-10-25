dataset="gsm8k"
inference_dir="./output/$dataset-inference"
inference_result_file="results.json"
reference_path="../lora-merging-datasets/test/$dataset.parquet"

report_name="report"


PYTHONPATH="." python ./run/eval.py \
  --dataset $dataset \
  --inference_dir $inference_dir \
  --inference_result_file $inference_result_file \
  --reference_path $reference_path \
  --report_name $report_name
