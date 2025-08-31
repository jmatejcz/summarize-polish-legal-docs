import torch
from loguru import logger
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import argparse

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from data_preprocess import create_train_data_for_prompt_tuning
from model_preparer import create_preparer, BaseModelPrepare


TRAIN_DOCUMENTS_PATH = "data/training/train/documents"
TRAIN_SUMMARIES_PATH = "data/training/train/summaries"


class QLoRADataset(Dataset):
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


class QLoRATrainer:
    """QLoRA trainer that uses model preparers for consistency"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.preparer = None
        self.model = None

    def _get_target_modules(self) -> list:
        """Get target modules based on model architecture"""
        if any(
            name in self.model_name.lower()
            for name in ["qwen", "gemma", "llama", "bielik"]
        ):
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

    def setup_model_and_lora(
        self, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1
    ):
        """Setup model with LoRA using preparer"""
        logger.info(f"Setting up model with LoRA: {self.model_name}")

        # Create preparer (this loads the base model with quantization)
        self.preparer = create_preparer(
            model_name=self.model_name, quantize=True, quantize_bits=4
        )

        # Enable gradient checkpointing if available
        if hasattr(self.preparer.model, "gradient_checkpointing_enable"):
            self.preparer.model.gradient_checkpointing_enable()

        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=self._get_target_modules(),
            bias="none",
        )

        # Apply LoRA to the preparer's model
        self.preparer.model = prepare_model_for_kbit_training(self.preparer.model)
        self.model = get_peft_model(self.preparer.model, lora_config)

        # Log parameters
        trainable_params, all_params = self.model.get_nb_trainable_parameters()
        logger.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)"
        )

    def train(
        self,
        output_dir: str,
        epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        max_length: int = 3072,
        seed: int = 42,
    ) -> str:
        """Train the model"""

        torch.manual_seed(seed)
        np.random.seed(seed)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("Starting training with config:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Epochs: {epochs}, LR: {learning_rate}")
        logger.info(f"  Output: {output_dir}")

        if self.model is None:
            raise RuntimeError("Model not setup. Call setup_model_and_lora() first.")

        logger.info("Loading training data...")
        train_data = create_train_data_for_prompt_tuning(
            documents_path=TRAIN_DOCUMENTS_PATH,
            target_path=TRAIN_SUMMARIES_PATH,
            max_len=max_length,
        )
        logger.info(f"Loaded {len(train_data)} training examples")

        # Create dataset using preparer
        train_dataset = QLoRADataset(train_data, self.preparer, max_length)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.preparer.tokenizer, mlm=False, pad_to_multiple_of=8
        )

        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.05,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=5,
            save_strategy="epoch",
            eval_strategy="no",
            save_total_limit=2,
            remove_unused_columns=False,
            seed=seed,
            fp16=True,
            dataloader_pin_memory=False,
            group_by_length=True,
            report_to=None,
            gradient_checkpointing=True,
            max_grad_norm=0.3,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        logger.info("Starting training...")
        train_result = trainer.train()

        logger.info("Saving model...")
        self.model.save_pretrained(output_path)
        self.preparer.tokenizer.save_pretrained(output_path)

        # Save training info
        training_info = {
            "model_name": self.model_name,
            "training_config": {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
            "training_metrics": train_result.metrics,
            "train_samples": len(train_data),
        }

        with open(output_path / "training_info.json", "w") as f:
            import json

            json.dump(training_info, f, indent=2)

        logger.info(f"Training completed! Model saved to: {output_path}")
        logger.info(
            f"Final training loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}"
        )

        return str(output_path)


def train_model(
    model_name: str,
    output_dir: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_length: int = 3072,
    seed: int = 42,
):
    """Train a single QLoRA model using preparers"""

    trainer = QLoRATrainer(model_name)

    # Setup model with LoRA
    trainer.setup_model_and_lora(
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
    )

    # Train the model
    return trainer.train(
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(description="QLoRA Training using Preparers")

    # Model and output
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Training parameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
