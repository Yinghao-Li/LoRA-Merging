"""
# Author: Yinghao Li
# Modified: October 22nd, 2024
# ---------------------------------------
# Description: Download and pre-process the GSM8K dataset
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
from seqlbtoolkit.io import set_logging, logging_args, progress_bar


logger = logging.getLogger(__name__)
accelerator = Accelerator()


@dataclass
class Arguments:
    output_dir: Optional[str] = field(
        default="../lora-merging-datasets",
        metadata={"help": "The output directory to save the dataset."},
    )


def main(args):
    splits = {"train": "main/train-00000-of-00001.parquet", "test": "main/test-00000-of-00001.parquet"}
    training_df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])
    test_df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["test"])

    output_dir = osp.abspath(args.output_dir)
    if not osp.isdir(output_dir):
        os.makedirs(output_dir)


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
