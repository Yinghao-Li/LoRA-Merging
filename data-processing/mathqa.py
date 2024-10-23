"""
# Author: Yinghao Li
# Modified: October 23rd, 2024
# ---------------------------------------
# Description: Download and pre-process the SVAMP dataset

Notice that the `equation` column is ignored
This dataset contains only 700 data points
"""

import os
import sys
import wget
import os.path as osp
import logging
import pandas as pd
import zipfile
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator

from transformers import HfArgumentParser
from seqlbtoolkit.io import set_logging, logging_args


logger = logging.getLogger(__name__)
accelerator = Accelerator()

DATASET_NAME = "mathqa"


@dataclass
class Arguments:
    output_dir: Optional[str] = field(
        default="../lora-merging-datasets",
        metadata={"help": "The output directory to save the dataset."},
    )


def reformat_instruction(row):
    instruction = f"{row['Problem'].strip()}\n\n"
    instruction += "Options:\n"
    instruction += row["options"].strip()

    return instruction


def reformat_response(row):
    response = f"{row['Rationale'].strip()[1:-1]}\n\n"
    response += f"So the answer is: {row['correct'].strip()}"

    return response


def main(args):
    url = "https://math-qa.github.io/math-QA/data/MathQA.zip"
    download_path = osp.join(args.output_dir, "tmp", "MathQA.zip")
    if not osp.exists(download_path):
        logger.info(f"Downloading the {DATASET_NAME} dataset to {download_path}...")
        os.makedirs(osp.dirname(download_path), exist_ok=True)
        wget.download(url, download_path)

    unzip_dir = osp.join(args.output_dir, "tmp", "MathQA")
    if not osp.exists(unzip_dir):
        logger.info(f"Unzipping the {DATASET_NAME} dataset to {unzip_dir}...")

        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(path=unzip_dir)

    training_df = pd.read_json(osp.join(unzip_dir, "train.json"))
    valid_df = pd.read_json(osp.join(unzip_dir, "dev.json"))
    test_df = pd.read_json(osp.join(unzip_dir, "test.json"))

    training_df["instruction"] = training_df.apply(reformat_instruction, axis=1)
    valid_df["instruction"] = valid_df.apply(reformat_instruction, axis=1)
    test_df["instruction"] = test_df.apply(reformat_instruction, axis=1)
    training_df["response"] = training_df.apply(reformat_response, axis=1)
    valid_df["response"] = valid_df.apply(reformat_response, axis=1)
    test_df["response"] = test_df.apply(reformat_response, axis=1)

    training_df = training_df[["instruction", "response", "category"]]
    valid_df = valid_df[["instruction", "response", "category"]]
    test_df = test_df[["instruction", "response", "category"]]

    training_output_dir = osp.join(args.output_dir, "train")
    os.makedirs(training_output_dir, exist_ok=True)

    valid_output_dir = osp.join(args.output_dir, "valid")
    os.makedirs(valid_output_dir, exist_ok=True)

    test_output_dir = osp.join(args.output_dir, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    logger.info(f"Saving the {DATASET_NAME} dataset to {args.output_dir}...")

    training_df.to_parquet(osp.join(training_output_dir, f"{DATASET_NAME}.parquet"))
    logger.info(f"Training dataset saved to {training_output_dir}")

    valid_df.to_parquet(osp.join(valid_output_dir, f"{DATASET_NAME}.parquet"))
    logger.info(f"Validation dataset saved to {valid_output_dir}")

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
