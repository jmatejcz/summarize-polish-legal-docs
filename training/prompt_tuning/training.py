import torch
from loguru import logger
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import argparse
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PromptTuningConfig, TaskType, get_peft_model
from data_preprocess import system_prompt, create_train_data_for_prompt_tuning

DOCUMENTS_PATH = "data/processed/documents/"
SUMMARIES_PATH = "data/processed/summaries/done/"


class PromptTuningDataset(Dataset):
    """Dataset specifically designed for prompt tuning"""

    def __init__(self, examples, tokenizer, max_length=4096):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example["input"]
        target_text = example["target"]

        # For prompt tuning, we create a simple completion format
        # The virtual tokens will be automatically prepended by the PEFT model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Streść poniższy dokument:\n{input_text}"},
            {"role": "assistant", "content": target_text},
        ]

        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # False because we include assistant response
        )
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Streść poniższy dokument:\n{input_text}"},
        ]

        prompt_only = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,  # True to get the assistant marker
        )

        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        prompt_tokens = self.tokenizer(
            prompt_only,
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


class PromptTuningPipeline:
    def __init__(
        self,
        model_name="Qwen/Qwen3-4B",
        output_dir="./output/prompt_tuning",
        num_virtual_tokens=20,
        torch_dtype="bfloat16",
        seed=42,
        use_quantization=True,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_virtual_tokens = num_virtual_tokens
        self.torch_dtype = (
            torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16
        )
        self.seed = seed
        self.use_quantization = use_quantization

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load tokenizer and model
        self._setup_model_and_tokenizer()

    def _setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer"""
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading model: {self.model_name}")

        # Setup quantization if enabled
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto",
        }

        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quantization_config

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable()

        # Configure prompt tuning
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=self.num_virtual_tokens,
            tokenizer_name_or_path=self.model_name,
        )

        # Apply prompt tuning to model
        self.model = get_peft_model(self.base_model, peft_config)

        # Log trainable parameters
        trainable_params, all_params = self.model.get_nb_trainable_parameters()
        logger.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)"
        )
        logger.info(f"Total parameters: {all_params:,}")

    def _prepare_datasets(self, val_split=0.1, max_length=3072):
        """Prepare training and validation datasets"""
        logger.info("Loading training data from documents and summaries")
        train_data = create_train_data_for_prompt_tuning(
            documents_path=DOCUMENTS_PATH,
            target_path=SUMMARIES_PATH,
            max_len=max_length,  # Reduced to leave room for virtual tokens and target
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

        train_dataset = PromptTuningDataset(train_examples, self.tokenizer, max_length)

        eval_dataset = None
        if len(eval_examples) > 0:
            eval_dataset = PromptTuningDataset(
                eval_examples, self.tokenizer, max_length
            )

        return train_dataset, eval_dataset

    def train(
        self,
        val_split=0.1,
        num_epochs=6,
        learning_rate=0.3,  # Higher LR for prompt tuning
        batch_size=2,
        gradient_accumulation_steps=4,
        max_length=3000,
        warmup_ratio=0.1,
        weight_decay=0.01,
    ):
        """Train the model with prompt tuning"""
        logger.info("Starting prompt tuning training")

        # Prepare datasets
        train_dataset, eval_dataset = self._prepare_datasets(
            val_split=val_split,
            max_length=max_length,
        )

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, pad_to_multiple_of=8
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
            bf16=self.torch_dtype == torch.bfloat16,
            fp16=self.torch_dtype == torch.float16,
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
        self.tokenizer.save_pretrained(self.output_dir)

        # Log and save metrics
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
        return self.model, trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Prompt Tuning Pipeline for Language Models"
    )

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
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        default=False,
        help="Enable 4-bit quantization for memory efficiency",
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

    # Other parameters
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_length", type=int, default=3000, help="Maximum sequence length"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Starting Prompt Tuning Pipeline with the following configuration:")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Virtual tokens: {args.num_virtual_tokens}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Use quantization: {args.use_quantization}")

    # Initialize the prompt tuning pipeline
    # pipeline = PromptTuningPipeline(
    #     model_name="Qwen/Qwen3-4B",
    #     output_dir="./output/qwen3_prompt_tuning_fixed/1",
    #     num_virtual_tokens=10,  # Start with fewer tokens
    #     use_quantization=True,
    # )

    pipeline = PromptTuningPipeline(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_virtual_tokens=args.num_virtual_tokens,
        use_quantization=args.use_quantization,
    )

    pipeline.train(
        num_epochs=4,  # Fewer epochs for prompt tuning
        learning_rate=0.1,  # High learning rate for prompt tuning
        batch_size=1,
        gradient_accumulation_steps=8,
        val_split=0.15,
    )


if __name__ == "__main__":
    main()
