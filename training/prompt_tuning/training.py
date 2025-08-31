import torch
from loguru import logger
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import argparse
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from peft import PromptTuningConfig, TaskType, get_peft_model
from data_preprocess import create_train_data_for_prompt_tuning
from evaluate_models import create_preparer, BaseModelPrepare


TRAIN_DOCUMENTS_PATH = "data/training/train/documents"
TRAIN_SUMMARIES_PATH = "data/training/train/summaries"


class PromptTuningDataset(Dataset):
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

        full_tokens = self.preparer.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        prompt_tokens = self.preparer.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        # Create labels - mask everything except the assistant's response
        labels = full_tokens.clone()
        labels[: len(prompt_tokens)] = -100  # Don't compute loss on prompt

        return {
            "input_ids": full_tokens,
            "attention_mask": torch.ones_like(full_tokens),
            "labels": labels,
        }


class PromptTuningTrainer:
    """Prompt tuning trainer that uses model preparers for consistency"""

    def __init__(
        self,
        model_name: str,
        output_dir: str = "./output/prompt_tuning",
        num_virtual_tokens: int = 20,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_virtual_tokens = num_virtual_tokens
        self.seed = seed
        self.preparer = None
        self.model = None

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def setup_model_and_prompt_tuning(self):
        """Setup model with prompt tuning using preparer"""
        logger.info(f"Setting up model with prompt tuning: {self.model_name}")

        # Create preparer (this loads the base model with quantization)
        self.preparer = create_preparer(
            model_name=self.model_name, quantize=True, quantize_bits=4
        )

        # Enable gradient checkpointing if available
        if hasattr(self.preparer.model, "gradient_checkpointing_enable"):
            self.preparer.model.gradient_checkpointing_enable()

        # Prompt tuning configuration
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=self.num_virtual_tokens,
            tokenizer_name_or_path=self.model_name,
        )

        # Apply prompt tuning to the preparer's model
        self.model = get_peft_model(self.preparer.model, peft_config)

        # Log parameters
        trainable_params, all_params = self.model.get_nb_trainable_parameters()
        logger.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)"
        )
        logger.info(f"Total parameters: {all_params:,}")

    def _prepare_datasets(self, val_split: float = 0.1, max_length: int = 3072):
        """Prepare training and validation datasets"""
        logger.info("Loading training data from documents and summaries")
        train_data = create_train_data_for_prompt_tuning(
            documents_path=TRAIN_DOCUMENTS_PATH,
            target_path=TRAIN_SUMMARIES_PATH,
            max_len=max_length,
        )
        logger.info(f"Loaded {len(train_data)} training examples")

        if val_split > 0:
            # Shuffle data
            import random

            random.seed(self.seed)
            random.shuffle(train_data)

            # Split data
            split_idx = int(len(train_data) * (1 - val_split))
            train_examples = train_data[:split_idx]
            eval_examples = train_data[split_idx:]
        else:
            train_examples = train_data
            eval_examples = []

        logger.info(
            f"Split data: {len(train_examples)} train, {len(eval_examples)} eval examples"
        )

        # Create datasets using preparer
        train_dataset = PromptTuningDataset(train_examples, self.preparer, max_length)
        eval_dataset = None
        if len(eval_examples) > 0:
            eval_dataset = PromptTuningDataset(eval_examples, self.preparer, max_length)

        return train_dataset, eval_dataset

    def train(
        self,
        val_split: float = 0.1,
        num_epochs: int = 6,
        learning_rate: float = 0.3,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        max_length: int = 3000,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
    ):
        """Train the model with prompt tuning"""

        if self.model is None:
            raise RuntimeError(
                "Model not setup. Call setup_model_and_prompt_tuning() first."
            )

        logger.info("Starting prompt tuning training")

        # Prepare datasets
        train_dataset, eval_dataset = self._prepare_datasets(
            val_split=val_split,
            max_length=max_length,
        )

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.preparer.tokenizer, mlm=False, pad_to_multiple_of=8
        )

        # Training arguments optimized for prompt tuning
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset else "no",
            logging_steps=10,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            remove_unused_columns=False,
            seed=self.seed,
            fp16=True,  # Use fp16 for memory efficiency
            dataloader_pin_memory=False,
            group_by_length=True,
            report_to=None,  # Disable wandb/tensorboard
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train model
        logger.info("Starting training")
        train_result = trainer.train()

        # Save model and metrics
        logger.info("Saving model")
        self.model.save_pretrained(self.output_dir)
        self.preparer.tokenizer.save_pretrained(self.output_dir)

        # Save training info
        training_info = {
            "model_name": self.model_name,
            "prompt_tuning_config": {
                "num_virtual_tokens": self.num_virtual_tokens,
            },
            "training_config": {
                "epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
            "training_metrics": train_result.metrics,
            "train_samples": len(train_dataset),
        }

        with open(Path(self.output_dir) / "training_info.json", "w") as f:
            import json

            json.dump(training_info, f, indent=2)

        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # Evaluate if we have validation data
        if eval_dataset:
            logger.info("Running evaluation")
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)

        logger.info("Training complete!")
        logger.info(f"Model saved to: {self.output_dir}")

        return self.model, trainer


def train_prompt_tuning_model(
    model_name: str,
    output_dir: str,
    num_virtual_tokens: int = 20,
    epochs: int = 6,
    learning_rate: float = 0.3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_length: int = 3000,
    val_split: float = 0.1,
    seed: int = 42,
) -> str:
    """Train a prompt tuning model using preparers"""

    trainer = PromptTuningTrainer(
        model_name=model_name,
        output_dir=output_dir,
        num_virtual_tokens=num_virtual_tokens,
        seed=seed,
    )

    # Setup model with prompt tuning
    trainer.setup_model_and_prompt_tuning()

    # Train the model
    trainer.train(
        val_split=val_split,
        num_epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
    )

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Prompt Tuning using Preparers")

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Name or path of the base model to fine-tune",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/prompt_tuning",
        help="Directory to save the trained model and outputs",
    )
    parser.add_argument(
        "--num_virtual_tokens",
        type=int,
        default=10,
        help="Number of virtual tokens for prompt tuning",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=4, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="Learning rate for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Per-device batch size"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of data to use for validation",
    )
    parser.add_argument(
        "--max_length", type=int, default=3000, help="Maximum sequence length"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    logger.info("Starting Prompt Tuning Pipeline with the following configuration:")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Virtual tokens: {args.num_virtual_tokens}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")

    # Train using the refactored approach
    train_prompt_tuning_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_virtual_tokens=args.num_virtual_tokens,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
