import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import sys
import os.path as osp
import logging
from accelerate import Accelerator
from transformers import HfArgumentParser, set_seed

from src.args import PipelineArguments, DataArguments, ModelArguments, TrainingArguments, argument_processing
from src.train import CausalLMTrainer as Trainer
from seqlbtoolkit.io import set_logging

logger = logging.getLogger(__name__)
accelerator = Accelerator()


def main(pipeline_args, model_args, data_args, training_args):
    argument_processing(pipeline_args, model_args, data_args, training_args)

    trainer = Trainer(
        pipeline_args.task,
        model_args,
        data_args,
        training_args,
        disable_training_end_model_saving=pipeline_args.disable_training_end_model_saving,
    )

    if pipeline_args.task == "train":
        trainer.train()
    elif pipeline_args.task == "infer":
        trainer.infer()
    else:
        logger.error(f"Invalid task: {pipeline_args.task}")

    return None


if __name__ == "__main__":
    # --- set up arguments ---
    parser = HfArgumentParser((PipelineArguments, ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        pipeline_args, model_args, data_args, training_args = parser.parse_json_file(json_file=osp.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        pipeline_args, model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=osp.abspath(sys.argv[1]), allow_extra_keys=True
        )
    else:
        pipeline_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_logging(level="INFO")
    set_seed(training_args.seed)

    main(pipeline_args, model_args, data_args, training_args)
