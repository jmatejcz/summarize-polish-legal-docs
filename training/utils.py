"""Common utilities for fine-tuning pipelines"""

import torch
from loguru import logger
import numpy as np
from torch.utils.data import Dataset

from model_preparer import BaseModelPrepare
from pathlib import Path
import json
from abc import ABC, abstractmethod

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import get_peft_model
from data_preprocess import create_train_data_for_prompt_tuning
from evaluate_models import create_preparer  # or model_preparer import create_preparer


TRAIN_DOCUMENTS_PATH = "data/training/train/documents"
TRAIN_SUMMARIES_PATH = "data/training/train/summaries"


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


class BasePEFTTrainer(ABC):
    """Base class for Parameter Efficient Fine Tuning trainers"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.preparer = None
        self.model = None

    def _setup_preparer(self):
        """Common preparer setup with quantization"""
        logger.info(f"Setting up model preparer: {self.model_name}")

        self.preparer = create_preparer(
            model_name=self.model_name, quantize=True, quantize_bits=4
        )

        # Enable gradient checkpointing if available
        if hasattr(self.preparer.model, "gradient_checkpointing_enable"):
            self.preparer.model.gradient_checkpointing_enable()

    @abstractmethod
    def _create_peft_config(self, **kwargs):
        """Create PEFT configuration - must be implemented by subclasses"""
        pass

    @abstractmethod
    def _prepare_model_for_peft(self):
        """Prepare model for PEFT - can be overridden by subclasses"""
        # Default implementation - just return the preparer's model
        return self.preparer.model

    def setup_model_and_peft(self, **peft_kwargs):
        """Setup model with PEFT technique"""
        # Setup preparer first
        self._setup_preparer()

        # Create PEFT configuration
        peft_config = self._create_peft_config(**peft_kwargs)

        # Prepare model for PEFT (e.g., k-bit training for LoRA)
        prepared_model = self._prepare_model_for_peft()

        # Apply PEFT to the model
        self.model = get_peft_model(prepared_model, peft_config)

        # Log parameters
        self._log_parameters()

    def _log_parameters(self):
        """Log trainable and total parameters"""
        trainable_params, all_params = self.model.get_nb_trainable_parameters()
        logger.info(
            f"Trainable parameters: {trainable_params:,} "
            f"({100 * trainable_params / all_params:.2f}%)"
        )
        logger.info(f"Total parameters: {all_params:,}")

    def _prepare_dataset(self, max_length: int):
        """Prepare training dataset"""
        logger.info("Loading training data from documents and summaries")
        train_data = create_train_data_for_prompt_tuning(
            documents_path=TRAIN_DOCUMENTS_PATH,
            target_path=TRAIN_SUMMARIES_PATH,
            max_len=max_length,
        )
        logger.info(f"Loaded {len(train_data)} training examples")

        # Create dataset using preparer
        train_dataset = BaseDataset(train_data, self.preparer, max_length)
        return train_dataset

    def _create_training_args(
        self,
        output_dir: str,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        gradient_accumulation_steps: int,
        seed: int,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        **kwargs,
    ):
        """Create training arguments with common settings"""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            eval_strategy="no",
            logging_steps=10,
            save_total_limit=3,
            load_best_model_at_end=False,
            remove_unused_columns=False,
            seed=seed,
            fp16=True,
            dataloader_pin_memory=False,
            group_by_length=True,
            report_to=None,
            **kwargs,  # Allow subclasses to add specific arguments
        )

    def _execute_training(self, train_dataset, training_args):
        """Execute the training loop"""
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.preparer.tokenizer, mlm=False, pad_to_multiple_of=8
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # Train model
        logger.info("Starting training")
        train_result = trainer.train()

        return trainer, train_result

    def _save_model_and_info(
        self,
        output_dir: str,
        train_result,
        training_config: dict,
        train_dataset_size: int,
        peft_config: dict = None,
    ):
        """Save model, tokenizer, and training information"""
        output_path = Path(output_dir)

        # Save model and tokenizer
        logger.info("Saving model")
        self.model.save_pretrained(output_path)
        self.preparer.tokenizer.save_pretrained(output_path)

        # Create training info
        training_info = {
            "model_name": self.model_name,
            "training_config": training_config,
            "training_metrics": train_result.metrics,
            "train_samples": train_dataset_size,
        }

        # Add PEFT-specific config if provided
        if peft_config:
            training_info["peft_config"] = peft_config

        # Save training info
        with open(output_path / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)

        logger.info("Training complete!")
        logger.info(f"Model saved to: {output_dir}")

        return training_info

    def train(
        self,
        output_dir: str,
        epochs: int = 6,
        learning_rate: float = 0.3,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        max_length: int = 3072,
        seed: int = 42,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        **training_kwargs,
    ):
        """Main training method"""

        if self.model is None:
            raise RuntimeError("Model not setup. Call setup_model_and_peft() first.")

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Starting PEFT training")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Output directory: {output_dir}")

        # Prepare dataset
        train_dataset = self._prepare_dataset(max_length=max_length)

        # Create training arguments
        training_args = self._create_training_args(
            output_dir=output_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            seed=seed,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            **training_kwargs,
        )

        # Execute training
        trainer, train_result = self._execute_training(train_dataset, training_args)

        # Prepare training config for saving
        training_config = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_length": max_length,
            "seed": seed,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
        }
        training_config.update(training_kwargs)

        # Save model and info
        training_info = self._save_model_and_info(
            output_dir=output_dir,
            train_result=train_result,
            training_config=training_config,
            train_dataset_size=len(train_dataset),
        )

        return self.model, trainer, training_info
