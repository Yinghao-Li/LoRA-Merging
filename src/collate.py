"""
# Author: Yinghao Li
# Modified: August 30th, 2024
# ---------------------------------------
# Description: Collate function for ChemBERTa.
"""

import torch
import logging

from dataclasses import dataclass

from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

from seqlbtoolkit.training.dataset import Batch

logger = logging.getLogger(__name__)


class Collator:
    """
    Collator class for ChemBERTa.

    The collator is responsible for combining instances into a batch for model processing.
    It handles token padding to ensure consistent tensor shapes across batches. It also constructs attention masks.

    Attributes
    ----------
    _pad_id : int
        The padding token ID.
    """

    def __init__(self, tokenizer):
        """
        Initialize the Collator.

        Parameters
        ----------
        config : object
            The configuration object which should have attributes 'task_type' and 'pretrained_model_name_or_path'.
        """
        self.tokenizer = tokenizer

    def __call__(self, instances: list[dict], *args, **kwargs) -> Batch:
        """
        Collate instances into a batch.

        Given a list of instances, this method pads the atom_ids to ensure all instances have the same length.
        Attention masks are generated to distinguish actual tokens from padding tokens. The resulting instances
        are combined into a batch.

        Parameters
        ----------
        instance_list : list
            A list of instances where each instance is expected to have atom_ids, labels (lbs), and masks.

        Returns
        -------
        Batch
            A batch containing padded atom_ids, attention masks, labels, and masks.
        """
        tks = self.tokenizer(instances, add_special_tokens=False, truncation=False, padding=True, return_tensors="pt")

        ids_batch = tks["input_ids"]
        masks_batch = tks["attention_mask"]

        return Batch(
            input_ids=ids_batch,
            attention_mask=masks_batch,
        )


class LMDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, assistant_label_only: bool = False):
        super().__init__(tokenizer, mlm=False)
        self.assistant_label_only = assistant_label_only

    def __call__(self, instances: list[dict], *args, **kwargs) -> Batch:
        assert len(instances) == 1, ValueError(
            "LMDataCollator (for grad calculation) only supports single instance collation."
        )

        super_instances = [
            {"input_ids": inst["input_ids"], "attention_mask": inst["attention_mask"]} for inst in instances
        ]
        super_batch = super().__call__(super_instances)
        instance_ids = [inst["instance_idx"] for inst in instances]

        # valid when batch size is 1. Otherwise need to consider left padding.
        labels = super_batch["labels"].clone()
        if self.assistant_label_only:
            len_instrs = [inst["len_instr"] for inst in instances]
            for lb_inst, len_instr in zip(labels, len_instrs):
                lb_inst[:len_instr] = -100

        return Batch(
            input_ids=super_batch["input_ids"],
            attention_mask=super_batch["attention_mask"],
            labels=labels,
            instance_ids=instance_ids,
        )


@dataclass
class DataCollatorForLanguageModeling:
    """
    Modified from transformers.DataCollatorForLanguageModeling.

    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = None
    assistant_label_only: bool = False

    def __call__(self, examples) -> dict:
        # Handle dict or lists with proper padding and conversion to tensor.
        super_examples = [
            {"input_ids": inst["input_ids"], "attention_mask": inst["attention_mask"]} for inst in examples
        ]

        batch = self.tokenizer.pad(super_examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)

        labels = batch["input_ids"].clone()

        if self.tokenizer.pad_token_id is not None:

            if self.assistant_label_only:
                len_instrs = [inst["len_instr"] for inst in examples]

                for lb_inst, len_instr in zip(labels, len_instrs):
                    non_a_index = torch.nonzero(lb_inst != self.tokenizer.pad_token_id, as_tuple=True)[0][0].item()
                    lb_inst[non_a_index : non_a_index + len_instr] = self.tokenizer.pad_token_id

            labels[labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels

        return batch
