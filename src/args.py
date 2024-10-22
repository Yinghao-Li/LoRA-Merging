"""
# Author: Yinghao Li
# Modified: August 30th, 2024
# ---------------------------------------
# Description: Self-defined arguments
"""

import os
from transformers import TrainingArguments  # Import for easy references
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from accelerate.logging import get_logger

logger = get_logger(__name__)
accelerator = Accelerator()

__all__ = ["TrainingArguments", "PipelineArguments", "ModelArguments", "DataArguments"]


@dataclass
class PipelineArguments:
    task: Optional[str] = field(
        default="train",
        metadata={"choices": ("train", "infer"), "help": "The task to run the pipeline."},
    )
    task_type: Optional[str] = field(
        default="causal_lm",
        metadata={"choices": ("seq_cls", "causal_lm"), "help": "The task type of the pipeline."},
    )
    project_name: Optional[str] = field(
        default=None,
        metadata={"help": "The project name used for logging with wandb."},
    )
    disable_training_end_model_saving: Optional[bool] = field(
        default=False,
        metadata={"help": "Disable saving the model at the end of training."},
    )


# Define and parse arguments.
@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    adapter_name: str = field(
        default=None,
        metadata={"help": "Name of the adapter to train."},
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    modules_to_save: Optional[str] = field(
        default=None,
        metadata={"help": "comma separated list of target modules to save"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    checkpoint_idx: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA checkpoint index to load. Set to negative number to disable."},
    )
    ft_folder_name: Optional[str] = field(
        default="ft",
        metadata={"help": "The folder name to save the fine-tuned model."},
    )
    moe_adapter_dirs: Optional[str] = field(
        default=None,
        metadata={"nargs": "*", "help": "Comma separated list of adapter names to train."},
    )
    disable_adapters: Optional[bool] = field(
        default=False,
        metadata={"help": "Disable training adapters."},
    )

    def __post_init__(self):
        if self.checkpoint_idx is not None and self.checkpoint_idx < 0:
            self.checkpoint_idx = None
        if not self.moe_adapter_dirs:
            self.moe_adapter_dirs = None
        if isinstance(self.moe_adapter_dirs, str):
            self.moe_adapter_dirs = [self.moe_adapter_dirs]


@dataclass
class DataArguments:
    local_dataset_dir: Optional[str] = field(default=None, metadata={"help": "the directory to local dataset"})
    clustered_dataset_dir: Optional[str] = field(default=None, metadata={"help": "the directory to clustered dataset"})
    cluster_idx: Optional[int] = field(default=None, metadata={"help": "the index of the cluster to use"})
    split_idx: Optional[int] = field(default=None, metadata={"help": "the index of the split to use"})
    subset_ids_path: Optional[str] = field(default=None, metadata={"help": "the path to the subset index file"})
    training_partition: Optional[str] = field(
        default=None,
        metadata={
            "choices": ("train", "valid", "test"),
            "help": "the partition name for training dataset if using Huggingface dataset",
        },
    )
    valid_partition: Optional[str] = field(
        default=None,
        metadata={
            "choices": ("train", "valid", "test"),
            "help": "the partition name for validation dataset if using Huggingface dataset",
        },
    )
    test_partition: Optional[str] = field(
        default=None,
        metadata={"help": "the partition name for test dataset if using Huggingface dataset"},
    )
    disable_dataset_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Disable dataset caching when `mapping` is called."},
    )
    disable_seq_length_filter: Optional[bool] = field(
        default=False,
        metadata={"help": "Disable filtering samples based on sequence length."},
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    dataset_num_proc: Optional[int] = field(default=8, metadata={"help": "the number of mapping processes"})
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    dataset_text_field: str = field(default="text", metadata={"help": "Dataset field to use as input text."})
    max_seq_length: Optional[int] = field(default=4096)
    max_new_tokens: Optional[int] = field(default=128)
    max_prompt_length: Optional[int] = field(default=4096)
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )
    training_subsample: float = field(
        default=0,
        metadata={"help": "The fraction or number of the training dataset to use for training."},
    )
    valid_subsample: float = field(
        default=0,
        metadata={"help": "The fraction or number of the validation dataset to use for validation."},
    )
    test_subsample: float = field(
        default=0,
        metadata={"help": "The fraction or number of the test dataset to use for testing."},
    )
    metric_file_name: Optional[str] = field(
        default="metrics.json",
        metadata={"help": "The file name to save the metrics."},
    )
    no_assistant_response: Optional[bool] = field(
        default=False,
        metadata={"help": "Remove assistant response during training."},
    )
    assistant_label_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Use assistant labels only for gradient calculation."},
    )

    def __post_init__(self):
        if self.training_subsample < 0 or self.valid_subsample < 0 or self.test_subsample < 0:
            raise ValueError("Subsample values must be greater than or equal to 0.")

        if self.training_subsample > 1:
            self.training_subsample = int(self.training_subsample)
        if self.valid_subsample > 1:
            self.valid_subsample = int(self.valid_subsample)
        if self.test_subsample > 1:
            self.test_subsample = int(self.test_subsample)

        if self.split_idx is not None and self.split_idx < 0:
            self.split_idx = None

        assert not (self.no_assistant_response and self.assistant_label_only), ValueError(
            "Only one of `no_assistant_response` and `assistant_label_only` can be True."
        )


def argument_processing(pipeline_args, model_args, data_args, training_args):

    if pipeline_args.project_name:
        os.environ["WANDB_PROJECT"] = pipeline_args.project_name

    if data_args.cluster_idx is not None:
        training_args.output_dir = os.path.join(
            training_args.output_dir, model_args.ft_folder_name, f"cluster-{data_args.cluster_idx}"
        )
        data_args.clustered_dataset_dir = os.path.join(
            data_args.clustered_dataset_dir, f"cluster-{data_args.cluster_idx}"
        )
        logger.warning(f"Will be using dataset: {data_args.clustered_dataset_dir}")

    logger.info(f"The output directory is {training_args.output_dir}")

    return None
