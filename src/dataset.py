"""
# Author: Yinghao Li
# Modified: October 24th, 2024
# ---------------------------------------
# Description: Dataset loading and pre-processing
"""

import os.path as osp
from accelerate.logging import get_logger
from datasets import load_dataset
from accelerate import Accelerator, PartialState

accelerator = Accelerator()

logger = get_logger(__name__)


class CausalLMDataset:
    def __init__(
        self,
        dataset_name: str,
        local_dataset_dir: str = None,
        partition: str = "train",
        subsample: int | float = 0,
        max_seq_length: int = 128,
        max_prompt_length: int = 128,
        disable_dataset_cache: bool = False,
        disable_seq_length_filter: bool = False,
        n_experts: int = 0,
        seed: int = 0,
        **kwargs,
    ) -> None:
        assert partition in ["train", "test"], ValueError("Invalid partition name!")
        self.ds = None

        self.ds_name = dataset_name
        self.local_ds_dir = local_dataset_dir

        self.disable_cache = disable_dataset_cache
        self.disable_seq_length_filter = disable_seq_length_filter
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length

        self.partition = partition
        self.subsample = subsample

        self.seed = seed

        self.subset_ids = None
        self.subset_name = None

        self.n_experts = n_experts

    def load(self) -> None:
        assert self.local_ds_dir, "You must provide a local dataset directory to load the dataset."

        ds_path = osp.join(self.local_ds_dir, self.partition, f"{self.ds_name}.parquet")
        logger.info(f"Loding dataset {ds_path}")
        self.ds = load_dataset("parquet", data_files=ds_path, split=self.partition)

        self.subsample_instances()

        return self

    def subsample_instances(self):

        ds = self.ds

        if self.subset_ids:
            logger.warning(f"Subsampling the dataset using subset ids.")
            idx_map = {idx: i for i, idx in enumerate(ds["idx"])}
            ids_to_keep = [idx_map[idx] for idx in self.subset_ids]
            self.ds = ds.select(ids_to_keep)

        else:
            sub = self.subsample
            if sub:
                n_subsample = int(len(ds) * sub if isinstance(sub, float) else sub)
                logger.warning(f"Subsampling the dataset to {n_subsample} instances.")
                self.ds = ds.shuffle(seed=self.seed).select(range(n_subsample))

        return self

    def process(self, tokenizer=None, dataset_text_field: str = "text", num_proc=8) -> "CausalLMDataset":
        assert tokenizer is not None, "You must provide a tokenizer to process the dataset."
        mapping_func = getattr(self, f"process_{self.ds_name.lower().replace('-', '_')}", self.process_default)
        if mapping_func is self.process_default:
            logger.warning(f"No custom processing function found for {self.ds_name}. Using default processing.")

        fn_kwargs = {"tokenizer": tokenizer, "dataset_text_field": dataset_text_field}

        with PartialState().local_main_process_first():
            load_cache = False if self.disable_cache and accelerator.is_local_main_process else None

            ds = self.ds.map(
                mapping_func,
                num_proc=num_proc,
                load_from_cache_file=load_cache,
                fn_kwargs=fn_kwargs,
            )
            len_before = len(ds)

            filter_fn = lambda x: x["keep_instance"] and x["n_tks"] < self.max_seq_length
            if self.disable_seq_length_filter:
                filter_fn = lambda x: x["keep_instance"]

            ds = ds.filter(
                filter_fn,
                num_proc=num_proc,
                load_from_cache_file=load_cache,
            )
            len_after = len(ds)

            ds = ds.remove_columns(["n_tks", "keep_instance"])

            self.ds = ds

        if len_before != len_after:
            logger.warning(
                f"Filtered {len_before - len_after} samples from the dataset due to exceeding max_seq_length."
            )

        return self

    def process_default(
        self,
        sample,
        tokenizer=None,
        dataset_text_field: str = "text",
        **kwargs,
    ):
        sample["keep_instance"] = True

        instr = sample["instruction"]
        resp = sample["response"]

        if self.partition == "train":
            msgs = [{"role": "user", "content": instr}, {"role": "assistant", "content": resp}]
            txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            if not txt.endswith(tokenizer.eos_token):
                txt += tokenizer.eos_token

        elif self.partition == "test":
            msgs = [{"role": "user", "content": instr}]
            txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        else:
            raise ValueError("Invalid partition name!")

        tokenized = tokenizer(txt, truncation=False)
        sample["n_tks"] = len(tokenized["input_ids"])
        sample[dataset_text_field] = txt

        return sample
