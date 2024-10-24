"""
# Author: Yinghao Li
# Modified: October 24th, 2024
# ---------------------------------------
# Description:

Causal LM Trainer including fine-tuning and inference
"""

import torch
import os
import os.path as osp
import copy
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import tqdm
from dataclasses import asdict
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .dataset import CausalLMDataset
from .collate import DataCollatorForLanguageModeling

from trl import SFTTrainer, is_xpu_available
from seqlbtoolkit.io import save_json, dumps_yaml, progress_bar

logger = get_logger(__name__)
accelerator = Accelerator()

__all__ = ["CausalLMTrainer"]


class CausalLMTrainer:

    def __init__(
        self,
        task,
        model_args,
        data_args,
        training_args,
        disable_training_end_model_saving=False,
    ):
        """Initializes the Trainer object."""

        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.disable_training_end_model_saving = disable_training_end_model_saving
        self.model = None
        self.tokenizer = None
        self.datasets = None
        self.trainer = None
        self.optimizer_state = None

        self.tokenizer_size_changed = False
        self.task = task

        self.lora_ckpt_dir = self.training_args.output_dir
        if self.model_args.checkpoint_idx is not None:
            self.lora_ckpt_dir = osp.join(self.lora_ckpt_dir, f"checkpoint-{self.model_args.checkpoint_idx}")

        if task == "train":
            self.prepare_for_finetuning()
        elif task == "infer":
            self.prepare_for_inference()
        elif task == "infer-vllm":
            self.prepare_for_inference_vllm()
        else:
            raise ValueError(f"Invalid task: {task}. Must be one of 'train', 'infer', 'embed'.")

    def prepare_for_finetuning(self):
        self.initialize_tokenizer()
        self.initialize_model()
        self.initialize_datasets()
        self.initialize_trainer()
        return self

    def prepare_for_inference(self):
        self.load_model_and_tokenizer()
        self.initialize_datasets()
        return self

    def prepare_for_inference_vllm(self):
        model_dir = self.model_args.model_name_or_path
        if self.training_args.output_dir:
            model_dir = self.training_args.output_dir

        self.model = LLM(
            model=self.model_args.model_name_or_path,
            enforce_eager=True,
            enable_lora=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.initialize_datasets()
        return self

    def initialize_model(self):

        # Step 1: Load the model

        bnb_config = None
        quant_storage_dtype = None
        device_map = "auto"

        if self.model_args.use_4bit_quantization:
            compute_dtype = getattr(torch, self.model_args.bnb_4bit_compute_dtype)
            quant_storage_dtype = getattr(torch, self.model_args.bnb_4bit_quant_storage_dtype)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.model_args.use_4bit_quantization,
                bnb_4bit_quant_type=self.model_args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.model_args.use_nested_quant,
                bnb_4bit_quant_storage=quant_storage_dtype,
            )

            if compute_dtype == torch.float16 and self.model_args.use_4bit_quantization:
                major, _ = torch.cuda.get_device_capability()
                if major >= 8 and accelerator.is_local_main_process:
                    logger.warning("=" * 80)
                    logger.warning("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                    logger.warning("=" * 80)

        elif self.model_args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=self.model_args.use_8bit_quantization)

        torch_dtype = (
            quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
        )

        if bnb_config:
            device_map = (
                {"": f"xpu:{Accelerator().process_index}"} if is_xpu_available() else {"": Accelerator().process_index}
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            use_cache=False,
        )

        if self.tokenizer_size_changed:
            # Resize the embeddings
            self.model.resize_token_embeddings(len(self.tokenizer))
            # Configure the pad token in the model
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.config.use_cache = not self.training_args.gradient_checkpointing

        return self

    def initialize_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            trust_remote_code=True,
            add_bos_token=False,
            add_eos_token=False,
            legacy=False,
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.tokenizer_size_changed = True
        self.tokenizer.padding_side = "right"

    def initialize_datasets(self):
        # Step 2: Load the dataset
        partition = "train"
        if "infer" in self.task:
            partition = "test"

        self.datasets = (
            CausalLMDataset(
                seed=self.training_args.seed,
                partition=partition,
                **asdict(self.data_args),
            )
            .load()
            .process(
                tokenizer=self.tokenizer,
                dataset_text_field=self.data_args.dataset_text_field,
                num_proc=self.data_args.dataset_num_proc,
            )
        )

        return self

    def initialize_trainer(self):
        # Step 3: Define the training arguments
        if self.training_args.gradient_checkpointing:
            self.training_args.gradient_checkpointing_kwargs = {"use_reentrant": self.model_args.use_reentrant}

        # Step 4: Define the LoraConfig
        peft_config = None

        if self.training_args.resume_from_checkpoint:
            # peft_config = PeftConfig.from_pretrained(self.training_args.resume_from_checkpoint)
            self.model = PeftModel.from_pretrained(
                self.model, self.training_args.resume_from_checkpoint, is_trainable=True
            )
            # peft_config = None

        elif self.model_args.use_peft_lora:
            peft_config = LoraConfig(
                lora_alpha=self.model_args.lora_alpha,
                lora_dropout=self.model_args.lora_dropout,
                r=self.model_args.lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=(
                    self.model_args.lora_target_modules.split(",")
                    if self.model_args.lora_target_modules != "all-linear"
                    else self.model_args.lora_target_modules
                ),
            )

        training_args = copy.deepcopy(self.training_args)
        training_args.remove_unused_columns = False
        # Step 5: Define the Trainer

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=self.datasets.ds,
            peft_config=peft_config,
            packing=False,
            dataset_text_field=self.data_args.dataset_text_field,
            dataset_num_proc=self.data_args.dataset_num_proc,
            max_seq_length=self.data_args.max_seq_length,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, assistant_label_only=self.data_args.assistant_label_only
            ),
        )
        if accelerator.is_local_main_process:
            try:
                self.trainer.model.print_trainable_parameters()
            except Exception as e:
                logger.warning(f"Failed to print trainable parameters: {e}")

        return None

    def train(self):
        """Runs the training process."""

        self.trainer.train(resume_from_checkpoint=None)

        if not self.disable_training_end_model_saving:
            self.save_model()

        return self

    def save_model(self):
        """
        We do not really care about the <pad> token embedding.
        therefore, we do not need to save the embedding layer explicitely.
        Otherwise, we should enable embedding and lm_head update using
        ```
        config = LoraConfig(
            r=64, lora_alpha=128, lora_dropout=0.0, target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"]
        )
        ```
        and the saving function will automatically save the embedding layer.

        Reference: https://huggingface.co/docs/peft/main/en/developer_guides/troubleshooting#extending-the-vocabulary
        """

        if accelerator.is_local_main_process:
            output_dir = self.training_args.output_dir

            self.trainer.save_model(output_dir)

        return None

    def load_model_and_tokenizer(self):

        if self.training_args.output_dir:

            logger.info("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
            logger.info("Initializing base model")
            self.initialize_model()

            if not self.model_args.disable_adapters:
                logger.info(f"Loading PEFT adapter from {self.lora_ckpt_dir}")
                self.model.load_adapter(self.lora_ckpt_dir)

        else:
            logger.info("Loading tokenizer and base model. No PEFT adapter applied.")
            self.initialize_tokenizer()
            self.initialize_model()

        self.tokenizer.padding_side = "left"

        return self

    def load_tokenizer_model_adapters(self):

        logger.info("Loading tokenizer")
        # The tokenizers from different adapters should be the same.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.moe_adapter_dirs[0])
        self.tokenizer.padding_side = "left"

        logger.info("Initializing base model")
        self.initialize_model()

        logger.info(f"Loading PEFT adapters")

        for adapter_dir in self.model_args.moe_adapter_dirs:
            logger.info(f"Loading adapter from {adapter_dir}")
            adapter_name = osp.basename(adapter_dir)
            self.model.load_adapter(adapter_dir, adapter_name=adapter_name)

        return self

    def load_optimizer(self):
        if osp.exists(self.lora_ckpt_dir):
            optimizer_path = osp.join(self.lora_ckpt_dir, "optimizer.pt")
            if osp.exists(optimizer_path):
                logger.info(f"Loading optimizer from {optimizer_path}")
                self.optimizer_state = torch.load(optimizer_path, map_location="cpu")["state"]

        else:
            raise ValueError("No output directory specified for loading the optimizer state.")
        return None

    def infer_vllm(self):
        logger.info(
            f"Initializing pipeline for text generation with temperature {self.model_args.inference_temperature}"
        )
        test_ds = self.datasets.ds

        logger.info("Generating responses...")

        outputs = self.model.generate(
            test_ds["text"],
            sampling_params=SamplingParams(
                temperature=self.model_args.inference_temperature,
                max_tokens=self.data_args.max_new_tokens,
            ),
            lora_request=LoRARequest(
                lora_name="default",
                lora_int_id=1,
                lora_path=self.training_args.output_dir,
            ),
        )

        logger.info("Saving results.")
        indexed_results = list()
        for idx, output in zip(test_ds["instance_idx"], outputs):
            generated_text = output.outputs[0].text
            indexed_results.append({"idx": idx, "generated": generated_text})

        output_dir = self.training_args.output_dir
        if self.data_args.inference_dir:
            output_dir = osp.join(output_dir, self.data_args.inference_dir)
        elif self.model_args.checkpoint_idx is not None:
            output_dir = osp.join(output_dir, f"checkpoint-{self.model_args.checkpoint_idx}")

        output_path = osp.join(output_dir, self.data_args.output_file_name)
        save_json(indexed_results, output_path)

        logger.info(f"Results saved to {output_path}.")

        return None

    def infer(self):
        pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.data_args.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        test_ds = self.datasets.test_ds

        test_loader = DataLoader(
            KeyDataset(test_ds, "text"),
            batch_size=self.training_args.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False,
        )
        try:
            test_loader = accelerator.prepare(test_loader)
        except Exception as e:
            logger.warning(f"Failed to prepare the test loader: {e}")

        results = list()
        logger.info("Generating responses...")
        # with progress_bar as pb:
        for batch in tqdm(test_loader):
            outputs = pipe(batch, batch_size=self.training_args.per_device_eval_batch_size, return_full_text=False)
            all_results = accelerator.gather_for_metrics(outputs, use_gather_object=True)
            results.extend(all_results)

        if accelerator.is_local_main_process:
            results = [output[0]["generated_text"] for output in results]
            indexed_results = [{"idx": idx, "generated": gen} for idx, gen in zip(test_ds["instance_idx"], results)]

            output_dir = self.training_args.output_dir
            if self.model_args.checkpoint_idx is not None:
                output_dir = osp.join(output_dir, f"checkpoint-{self.model_args.checkpoint_idx}")

            metric_path = osp.join(output_dir, self.data_args.metric_file_name)

            save_json(indexed_results, metric_path)
            logger.info(f"Results saved to {metric_path}.")

        return None
