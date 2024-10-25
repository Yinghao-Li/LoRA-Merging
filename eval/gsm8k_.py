"""
# Author: Yinghao Li
# Modified: October 25th, 2024
# ---------------------------------------
# Description: Evaluate the GSM8K results
"""

import sys
import re
import os.path as osp
import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator

from transformers import HfArgumentParser
from seqlbtoolkit.io import set_logging, logging_args, save_yaml, dumps_yaml


logger = logging.getLogger(__name__)
accelerator = Accelerator()

DATASET_NAME = "gsm8k"


@dataclass
class Arguments:
    inference_dir: Optional[str] = field(
        metadata={"help": "The directory containing the evaluation results."},
    )
    reference_path: Optional[str] = field(
        metadata={"help": "The output directory to save the dataset."},
    )


def main(args):
    logger.info(f"Evaluating {DATASET_NAME}...")

    logger.info(f"Loading inference results from {args.inference_dir}...")
    inference_df = pd.read_json(osp.join(args.inference_dir, "results.json"))
    logger.info(f"Loaded {len(inference_df)} results.")

    logger.info(f"Loading reference data from {args.reference_path}...")
    reference_df = pd.read_parquet(args.reference_path)
    logger.info(f"Loaded {len(reference_df)} reference data.")

    df = pd.merge(inference_df[["idx", "generated"]], reference_df[["idx", "response"]], on="idx", how="inner")

    logger.info(f"Calculating the metrics score...")

    missing_ids = sorted(list(set(inference_df["idx"]) - set(df["idx"])))

    n_correct = 0
    correct_ids = []
    incorrect_ids = []
    for idx, pred_str, ref_str in zip(df["idx"], df["generated"], df["response"]):
        try:
            pred = re.search(r"####\s*(.+)", pred_str).group(1).strip()
        except AttributeError:
            pred = "missing"

        ref = re.search(r"####\s*(.+)", ref_str).group(1).strip()

        if pred == ref:
            correct_ids.append(idx)
            n_correct += 1
        else:
            incorrect_ids.append(idx)

    accuracy = n_correct / len(df)

    report = {
        "accuracy": accuracy,
        "missing_ids": missing_ids,
        "incorrect_ids": incorrect_ids,
        "correct_ids": correct_ids,
    }
    logger.info("Results:")
    logger.info(dumps_yaml({"accuracy": accuracy, "n_missing": len(missing_ids)}))

    save_yaml(report, osp.join(args.inference_dir, "report.yaml"))

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
