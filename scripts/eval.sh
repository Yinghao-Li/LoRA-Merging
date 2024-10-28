dataset="mathqa"
inference_dir="./output/$dataset-inference"
reference_path="../lora-merging-datasets/test/$dataset.parquet"

inference_result_file="results.json"
# inference_result_file="results-no-adapters.json"

report_name="report"
# report_name="report-no-adapters"


PYTHONPATH="." python ./run/eval.py \
  --dataset $dataset \
  --inference_dir $inference_dir \
  --inference_result_file $inference_result_file \
  --reference_path $reference_path \
  --report_name $report_name
