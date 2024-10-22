import yaml
import json
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
        clustered_dataset_dir: str = None,
        split_idx: int = None,
        subset_ids_path: str = None,
        training_partition: str = None,
        valid_partition: str = None,
        test_partition: str = None,
        training_subsample: int | float = 0,
        valid_subsample: int | float = 0,
        test_subsample: int | float = 0,
        max_seq_length: int = 128,
        max_prompt_length: int = 128,
        disable_dataset_cache: bool = False,
        disable_seq_length_filter: bool = False,
        no_assistant_response: bool = False,
        n_experts: int = 0,
        seed: int = 0,
        **kwargs,
    ) -> None:

        self.ds_name = dataset_name
        self.local_ds_dir = local_dataset_dir
        self.clustered_ds_dir = clustered_dataset_dir
        self.split_idx = split_idx
        self.subset_ids_path = subset_ids_path

        self.disable_cache = disable_dataset_cache
        self.disable_seq_length_filter = disable_seq_length_filter
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length

        self.training_ds = None
        self.valid_ds = None
        self.test_ds = None

        self.training_partition = training_partition
        self.valid_partition = valid_partition
        self.test_partition = test_partition

        self.meta = None

        self.training_sub = training_subsample
        self.valid_sub = valid_subsample
        self.test_sub = test_subsample

        self.seed = seed
        self.no_assistant_response = no_assistant_response

        self.subset_ids = None
        self.subset_name = None

        self.n_experts = n_experts

    def load(self) -> None:
        if self.clustered_ds_dir:
            ds_dir = self.clustered_ds_dir
            assert self.split_idx is None

        else:
            assert self.local_ds_dir, "You must provide a local dataset directory to load the dataset."
            ds_dir = osp.join(self.local_ds_dir, self.ds_name)

            if self.split_idx is not None:
                ds_dir = osp.join(ds_dir, f"split-{self.split_idx}")

        meta_path = osp.join(ds_dir, "meta.yaml")

        if osp.exists(meta_path):
            with open(meta_path, "r") as f:
                self.meta = yaml.safe_load(f)

        logger.info(f"Loding dataset from {ds_dir}")

        self.load_(ds_dir=ds_dir)
        self.load_subset_ids()

        self.subsample()

        return self

    def load_(self, ds_dir: str, **kwargs) -> None:
        """
        Load the dataset from the downloaded Huggingface repository.
        """

        for prt in ["training", "valid", "test"]:
            partition = getattr(self, f"{prt}_partition", None)
            if partition:
                if partition == "valid":
                    partition = "validation"
                try:
                    setattr(self, f"{prt}_ds", load_dataset(ds_dir, split=partition, **kwargs))
                except ValueError:
                    logger.warning(f"{prt.capitalize()} split is supposed to be loaded, but the file is not found.")

        return self

    def load_subset_ids(self) -> None:
        if self.subset_ids_path and osp.exists(self.subset_ids_path):
            with open(self.subset_ids_path, "r") as f:
                subset = json.load(f)
            self.subset_ids = subset["ids"]
            self.subset_name = subset["subset_name"]

        elif self.subset_ids_path:
            logger.warning(f"Subset ids file not found at {self.subset_ids_path}.")
            self.subset_ids = None
            self.subset_name = None

        return None

    def subsample(self):

        for prt in ["training", "valid", "test"]:
            ds = getattr(self, f"{prt}_ds", None)
            if ds:
                if self.subset_ids:
                    logger.warning(f"Subsampling {prt} dataset using subset ids.")
                    idx_map = {idx: i for i, idx in enumerate(ds["idx"])}
                    ids_to_keep = [idx_map[idx] for idx in self.subset_ids]
                    setattr(self, f"{prt}_ds", ds.select(ids_to_keep))
                else:
                    sub = getattr(self, f"{prt}_sub", None)
                    if sub:
                        n_subsample = int(len(ds) * sub if isinstance(sub, float) else sub)
                        logger.warning(f"Subsampling {prt} dataset to {n_subsample} instances.")
                        setattr(self, f"{prt}_ds", ds.shuffle(seed=self.seed).select(range(n_subsample)))

        return self

    def process(self, tokenizer=None, dataset_text_field: str = "text", num_proc=8) -> "CausalLMDataset":
        assert tokenizer is not None, "You must provide a tokenizer to process the dataset."
        mapping_func = getattr(self, f"process_{self.ds_name.lower().replace('-', '_')}", None)
        if not mapping_func:
            return self

        fn_kwargs = {"tokenizer": tokenizer, "dataset_text_field": dataset_text_field}

        for attr in ["training_ds", "valid_ds", "test_ds"]:
            if not getattr(self, attr):
                continue

            with PartialState().local_main_process_first():
                columns_to_remove = list(set(getattr(self, attr).features) - set(["instance_idx", dataset_text_field]))
                load_cache = False if self.disable_cache and accelerator.is_local_main_process else None

                ds = getattr(self, attr).map(
                    mapping_func,
                    num_proc=num_proc,
                    load_from_cache_file=load_cache,
                    fn_kwargs=fn_kwargs | {"partition": attr.split("_")[0]},
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

                ds = ds.remove_columns(columns_to_remove + ["n_tks", "keep_instance"])

                setattr(self, attr, ds)

            if len_before != len_after:
                logger.warning(
                    f"Filtered {len_before - len_after} samples from {attr} dataset due to exceeding max_seq_length."
                )

        return self

    def process_bbh(
        self,
        sample,
        tokenizer=None,
        dataset_text_field: str = "text",
        partition: str = "test",
    ):
        assert partition != "train", "This function is only for validation or test partition!"

        instance_idx = sample["instance_id"]
        question = sample["question"]
        answer = sample["answer"]
        src = sample["src"]

        meta = self.meta["Promptings"][src]
        instr_txt = f"{meta['instruction']}\n\n"

        # disable examples for grad collection
        examples = meta["examples"]
        if self.for_grad_collection:
            examples = list()

        for example_idx, example in enumerate(examples):
            instr_txt += f"Example {example_idx + 1}:\n"
            instr_txt += f"{example}\n\n"
        instr_txt += f"Question:\nQ: {question}\n"

        prompt_msgs = [{"role": "user", "content": instr_txt}]

        prompt_txt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        prompt_tokenized = tokenizer(prompt_txt, truncation=False)

        while len(prompt_tokenized["input_ids"]) > self.max_prompt_length:
            if not examples:
                logger.warning(f"Prompt exceeds max_prompt_length for instance {instance_idx}.")
                break

            examples.pop()
            instr_txt = f"{meta['instruction']}\n\n"
            for example_idx, example in enumerate(meta["examples"]):
                instr_txt += f"Example {example_idx + 1}:\n"
                instr_txt += f"{example}\n\n"
            instr_txt += f"Question:\nQ: {question}\n"

            prompt_msgs = [{"role": "user", "content": instr_txt}]

            prompt_txt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            prompt_tokenized = tokenizer(prompt_txt, truncation=False)

        if partition == "test":
            msgs = [{"role": "user", "content": instr_txt}]
        else:
            msgs = [{"role": "user", "content": instr_txt}, {"role": "assistant", "content": answer}]

        if self.no_assistant_response:
            msgs = msgs[:-1] if msgs[-1]["role"] == "assistant" else msgs

        apply_generation_prompt = partition == "test" and not self.for_grad_collection
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=apply_generation_prompt)
        tokenized = tokenizer(txt, truncation=False)
        sample["n_tks"] = len(tokenized["input_ids"])

        if self.for_grad_collection:
            sample["input_ids"] = tokenized["input_ids"]
            sample["attention_mask"] = tokenized["attention_mask"]
            sample["instance_idx"] = instance_idx
            return sample

        sample[dataset_text_field] = txt

        sample["instance_idx"] = instance_idx

        return sample

    def process_mmlu(
        self,
        sample,
        tokenizer=None,
        dataset_text_field: str = "text",
        partition: str = "test",
    ):
        assert partition != "train", "This function is only for validation or test partition!"

        instance_idx = sample["instance_id"]
        question = sample["question"]
        answer = sample["answer"]
        src = sample["src"]

        opa = sample["OptionA"]
        opb = sample["OptionB"]
        opc = sample["OptionC"]
        opd = sample["OptionD"]

        meta = self.meta["Promptings"][src.replace("_test", "_dev")]
        instr_txt = f"Please solve the following multi-choice problems.\n\n"

        # disable examples for grad collection
        examples = meta["examples"]
        if self.for_grad_collection:
            examples = list()

        for example_idx, example in enumerate(examples):
            instr_txt += f"Example {example_idx + 1}:\n"
            instr_txt += f"{example['question']}\n\n"
            instr_txt += f"Option A: {example['OptionA']}\n"
            instr_txt += f"Option B: {example['OptionB']}\n"
            instr_txt += f"Option C: {example['OptionC']}\n"
            instr_txt += f"Option D: {example['OptionD']}\n\n"
            instr_txt += f"Answer: {example['answer']}\n\n"

        instr_txt += f"Question:\n\n"
        instr_txt += f"{question}\n\n"
        instr_txt += f"Option A: {opa}\nOption B: {opb}\nOption C: {opc}\nOption D: {opd}\n"

        prompt_msgs = [{"role": "user", "content": instr_txt}]

        prompt_txt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        prompt_tokenized = tokenizer(prompt_txt, truncation=False)

        while len(prompt_tokenized["input_ids"]) > self.max_prompt_length:
            if not examples:
                logger.warning(f"Prompt exceeds max_prompt_length for instance {instance_idx}.")
                break

            examples.pop()
            instr_txt = f"Please solve the following multi-choice problems.\n\n"
            for example_idx, example in enumerate(meta["examples"]):
                instr_txt += f"Example {example_idx + 1}:\n"
                instr_txt += f"{example['question']}\n\n"
                instr_txt += f"Option A: {example['OptionA']}\n"
                instr_txt += f"Option B: {example['OptionB']}\n"
                instr_txt += f"Option C: {example['OptionC']}\n"
                instr_txt += f"Option D: {example['OptionD']}\n\n"
                instr_txt += f"Answer: {example['answer']}\n\n"

            instr_txt += f"Question:\n\n"
            instr_txt += f"{question}\n\n"
            instr_txt += f"Option A: {opa}\nOption B: {opb}\nOption C: {opc}\nOption D: {opd}\n"

            prompt_msgs = [{"role": "user", "content": instr_txt}]

            prompt_txt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            prompt_tokenized = tokenizer(prompt_txt, truncation=False)

        if partition == "test":
            msgs = [{"role": "user", "content": instr_txt}]
        else:
            msgs = [{"role": "user", "content": instr_txt}, {"role": "assistant", "content": answer}]

        if self.no_assistant_response:
            msgs = msgs[:-1] if msgs[-1]["role"] == "assistant" else msgs

        apply_generation_prompt = partition == "test" and not self.for_grad_collection
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=apply_generation_prompt)
        tokenized = tokenizer(txt, truncation=False)
        sample["n_tks"] = len(tokenized["input_ids"])

        sample["instance_idx"] = instance_idx

        if self.for_grad_collection:
            sample["input_ids"] = tokenized["input_ids"]
            sample["attention_mask"] = tokenized["attention_mask"]
            return sample

        sample[dataset_text_field] = txt

        return sample

    def process_gsm8k(
        self,
        sample,
        tokenizer=None,
        dataset_text_field: str = "text",
        partition: str = "train",
        **kwargs,
    ):
        sample["keep_instance"] = True

        question = sample["question"]
        answer = sample["answer"]

        msgs = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        msgs_inst_only = msgs[:-1] if msgs[-1]["role"] == "assistant" else msgs
        if self.no_assistant_response:
            msgs = msgs_inst_only

        full_conv_txt = tokenizer.apply_chat_template(msgs, tokenize=False)
        instr_only_txt = tokenizer.apply_chat_template(msgs_inst_only, tokenize=False)

        if not full_conv_txt.endswith(tokenizer.eos_token):
            full_conv_txt += tokenizer.eos_token

        tokenized = tokenizer(full_conv_txt, truncation=False)
        sample["n_tks"] = len(tokenized["input_ids"])
        sample["len_instr"] = len(tokenizer(instr_only_txt, truncation=False)["input_ids"])

        if "train" in partition:
            sample[dataset_text_field] = full_conv_txt
            return sample

        instr_txt = tokenizer.apply_chat_template(msgs_inst_only, tokenize=False, add_generation_prompt=True)
        sample[dataset_text_field] = instr_txt

        return sample

    def process_math(
        self,
        sample,
        tokenizer=None,
        dataset_text_field: str = "text",
        partition: str = "train",
        **kwargs,
    ):
        return self.process_gsm8k(sample, tokenizer, dataset_text_field, partition, **kwargs)
