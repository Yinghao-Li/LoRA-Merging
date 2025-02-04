"""
# Author: Yinghao Li
# Modified: October 25th, 2024
# ---------------------------------------
# Description: Download and pre-process the SVAMP dataset

Notice that the `equation` column is ignored
This dataset contains only 700 data points
"""

import os
import sys
import os.path as osp
import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator

from transformers import HfArgumentParser
from seqlbtoolkit.io import set_logging, logging_args


logger = logging.getLogger(__name__)
accelerator = Accelerator()

DATASET_NAME = "svamp"


@dataclass
class Arguments:
    output_dir: Optional[str] = field(
        default="../lora-merging-datasets",
        metadata={"help": "The output directory to save the dataset."},
    )


def main(args):
    logger.info(f"Downloading and processing the {DATASET_NAME} dataset...")
    splits = {"train": "data/train-00000-of-00001.parquet", "test": "data/test-00000-of-00001.parquet"}
    training_df = pd.read_parquet("hf://datasets/ChilleD/SVAMP/" + splits["train"])
    test_df = pd.read_parquet("hf://datasets/ChilleD/SVAMP/" + splits["test"])

    training_df = training_df[["question_concat", "Answer", "Type"]]

    training_df.rename(
        columns={"question_concat": "instruction", "Answer": "response", "Type": "category"}, inplace=True
    )
    test_df.rename(columns={"question_concat": "instruction", "Answer": "response", "Type": "category"}, inplace=True)

    training_df["idx"] = [f"svamp.{cat}.train.{idx}" for idx, cat in enumerate(training_df["category"])]
    test_df["idx"] = [f"svamp.{cat}.test.{idx}" for idx, cat in enumerate(test_df["category"])]

    training_output_dir = osp.join(args.output_dir, "train")
    os.makedirs(training_output_dir, exist_ok=True)

    test_output_dir = osp.join(args.output_dir, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    logger.info(f"Saving the {DATASET_NAME} dataset to {args.output_dir}...")

    training_df.to_parquet(osp.join(training_output_dir, f"{DATASET_NAME}.parquet"))
    logger.info(f"Training dataset saved to {training_output_dir}")

    test_df.to_parquet(osp.join(test_output_dir, f"{DATASET_NAME}.parquet"))
    logger.info(f"Test dataset saved to {test_output_dir}")

    return None


if __name__ == "__main__":
    # --- set up arguments ---
    parser = HfArgumentParser((Arguments,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (arguments,) = parser.parse_json_file(json_file=osp.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        (arguments,) = parser.parse_yaml_file(yaml_file=osp.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        (arguments,) = parser.parse_args_into_dataclasses()

    set_logging(level="INFO")
    logging_args(arguments)

    main(arguments)
