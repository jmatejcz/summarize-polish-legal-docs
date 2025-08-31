"""Common utilities for fine-tuning pipelines"""

import torch
from loguru import logger
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Gemma3ForCausalLM,
    BitsAndBytesConfig,
)

from model_preparer import BaseModelPrepare


def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def setup_quantization_config():
    """Setup standard quantization configuration"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_base_model_and_tokenizer(
    model_name, torch_dtype=torch.float16, use_quantization=True
):
    """Load base model and tokenizer with common configuration"""
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {model_name}")

    model_args = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }

    if use_quantization:
        model_args["quantization_config"] = setup_quantization_config()

    # Load appropriate model class
    is_gemma = "gemma" in model_name.lower()
    if is_gemma:
        model_args["attn_implementation"] = "eager"  # or "flash_attention_2"
        base_model = Gemma3ForCausalLM.from_pretrained(model_name, **model_args)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    return base_model, tokenizer, is_gemma


def get_lora_target_modules(model_name):
    """Get appropriate target modules for LoRA based on model type"""
    if any(name in model_name.lower() for name in ["qwen", "gemma", "llama", "bielik"]):
        return [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    else:
        return ["q_proj", "v_proj", "k_proj", "o_proj"]


def log_trainable_parameters(model):
    """Log information about trainable parameters"""
    trainable_params, all_params = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)"
    )
    logger.info(f"Total parameters: {all_params:,}")


class BaseDataset(Dataset):
    """Dataset class that uses preparers for consistent chat formatting"""

    def __init__(self, examples, preparer: BaseModelPrepare, max_length=4096):
        self.examples = examples
        self.preparer = preparer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example["input"]
        target_text = example["target"]

        messages = self.preparer.create_training_chat_messages(input_text, target_text)
        prompt_messages = self.preparer._create_chat_messages(input_text)

        full_text = self.preparer._apply_chat_template(messages)
        prompt_text = self.preparer._apply_chat_template(prompt_messages)

        # Tokenize both
        full_tokens = self.preparer.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        prompt_tokens = self.preparer.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        input_ids = full_tokens["input_ids"].squeeze(0)
        attention_mask = full_tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = prompt_tokens["input_ids"].shape[-1]

        # Mask prompt tokens (only train on assistant response)
        if prompt_len < len(labels):
            labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def prepare_datasets(
    train_data,
    dataset_class,
    tokenizer,
    max_length,
    system_prompt,
    val_split=0.1,
    seed=42,
    **kwargs,
):
    """Common dataset preparation logic"""
    if val_split > 0:
        import random

        random.seed(seed)
        random.shuffle(train_data)

        split_idx = int(len(train_data) * (1 - val_split))
        train_examples = train_data[:split_idx]
        eval_examples = train_data[split_idx:]
    else:
        train_examples = train_data
        eval_examples = []

    logger.info(
        f"Split data: {len(train_examples)} train, {len(eval_examples)} eval examples"
    )

    train_dataset = dataset_class(
        train_examples, tokenizer, max_length, system_prompt, **kwargs
    )

    eval_dataset = None
    if len(eval_examples) > 0:
        eval_dataset = dataset_class(
            eval_examples, tokenizer, max_length, system_prompt, **kwargs
        )

    return train_dataset, eval_dataset


def save_training_info(output_path, model_name, config, metrics, train_samples):
    """Save training information to JSON file"""
    import json

    training_info = {
        "model_name": model_name,
        "config": config,
        "training_metrics": metrics,
        "train_samples": train_samples,
    }

    with open(output_path / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
