"""
# Author: Yinghao Li
# Modified: October 25th, 2024
# ---------------------------------------
# Description: Evaluate the generated results
"""

import sys
import os.path as osp
import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator

from transformers import HfArgumentParser
from seqlbtoolkit.io import set_logging, logging_args, save_yaml, dumps_yaml

from src.eval_funcs import EvalFuncs


logger = logging.getLogger(__name__)
accelerator = Accelerator()


@dataclass
class Arguments:
    inference_dir: Optional[str] = field(
        metadata={"help": "The directory containing the evaluation results."},
    )
    reference_path: Optional[str] = field(
        metadata={"help": "The output directory to save the dataset."},
    )
    inference_result_file: Optional[str] = field(
        default="results.json",
        metadata={"help": "The file containing the evaluation results."},
    )
    report_name: Optional[str] = field(
        default="report",
        metadata={"help": "The name of the report file."},
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset name."},
    )


def main(args):
    if args.dataset is not None:
        logger.info(f"Evaluating {args.dataset}...")
    else:
        logger.info(f"Evaluating...")

    logger.info(f"Loading inference results from {args.inference_dir}...")
    inference_df = pd.read_json(osp.join(args.inference_dir, args.inference_result_file))
    logger.info(f"Loaded {len(inference_df)} results.")

    logger.info(f"Loading reference data from {args.reference_path}...")
    reference_df = pd.read_parquet(args.reference_path)
    logger.info(f"Loaded {len(reference_df)} reference data.")

    df = pd.merge(inference_df[["idx", "generated"]], reference_df[["idx", "response"]], on="idx", how="inner")

    logger.info(f"Calculating the metrics score...")

    missing_ids = sorted(list(set(inference_df["idx"]) - set(df["idx"])))

    eval_func = getattr(EvalFuncs, args.dataset)
    report = eval_func(df)
    report["missing_ids"] = missing_ids

    logger.info("Results:")
    logger.info(dumps_yaml({"metrics": report["metrics"], "n_missing": len(missing_ids)}))

    save_yaml(report, osp.join(args.inference_dir, f"{args.report_name}.yaml"))

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
