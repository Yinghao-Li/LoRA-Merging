"""
# Author: Yinghao Li
# Modified: October 29th, 2024
# ---------------------------------------
# Description: Download and pre-process the MATH dataset
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

DATASET_NAME = "omnimath"


@dataclass
class Arguments:
    output_dir: Optional[str] = field(
        default="../lora-merging-datasets",
        metadata={"help": "The output directory to save the dataset."},
    )


def get_category(row):
    domain = row["domain"][0]
    cat = domain.split("->")[1]
    cat = cat.lower().strip()
    return cat


def main(args):
    logger.info(f"Processing the {DATASET_NAME} dataset...")

    test_df = pd.read_json("hf://datasets/KbsdJames/Omni-MATH/test.jsonl", lines=True)

    test_df.rename(columns={"problem": "instruction", "solution": "response"}, inplace=True)
    test_df = test_df[test_df["domain"].apply(lambda x: len(x) > 0)]

    test_df["category"] = test_df.apply(get_category, axis=1)

    test_df["idx"] = [f"{DATASET_NAME}.{cat}.test.{idx}" for (idx, cat) in enumerate(test_df["category"])]

    test_output_dir = osp.join(args.output_dir, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    logger.info(f"Saving the {DATASET_NAME} dataset to {args.output_dir}...")

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
