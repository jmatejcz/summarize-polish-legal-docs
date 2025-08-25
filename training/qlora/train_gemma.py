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
    Gemma3ForCausalLM,  # Use Gemma3 specific model
    BitsAndBytesConfig,
)

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from data_preprocess import system_prompt, create_train_data_for_prompt_tuning

DOCUMENTS_PATH = "data/processed/documents/"
SUMMARIES_PATH = "data/processed/summaries/done/"


class QLoRADataset(Dataset):
    """Dataset for QLoRA training with proper chat template format for Gemma3"""

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

        # Use Gemma3 chat template format
        messages = [
            {
                "role": "user",
                "content": f"{system_prompt}\n\nStreść poniższy dokument:\n{input_text}",
            },
            {"role": "assistant", "content": target_text},
        ]

        # Apply chat template for full conversation
        full_tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False,
        ).squeeze(0)

        # Get prompt part (without assistant response) for masking
        prompt_messages = [
            {
                "role": "user",
                "content": f"{system_prompt}\n\nStreść poniższy dokument:\n{input_text}",
            }
        ]

        prompt_tokens = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,  # Include assistant marker
        ).squeeze(0)

        # Truncate if needed
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[: self.max_length]

        # Create attention mask and labels
        attention_mask = torch.ones_like(full_tokens)
        labels = full_tokens.clone()

        # Mask prompt tokens so loss is only computed on assistant response
        if len(prompt_tokens) < len(full_tokens):
            labels[: len(prompt_tokens)] = -100

        return {
            "input_ids": full_tokens,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class QLoRAPipeline:
    def __init__(
        self,
        model_name="google/gemma-3-4b-it",
        output_dir="./output/qlora_training",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        seed=42,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.seed = seed

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load tokenizer and model
        self._setup_model_and_tokenizer()

    def _get_target_modules(self):
        """Get target modules for LoRA based on Gemma3 architecture"""
        if "gemma" in self.model_name.lower():
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
            # Default attention modules
            return ["q_proj", "v_proj", "k_proj", "o_proj"]

    def _setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer with QLoRA configuration for Gemma3"""
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Ensure pad token is set (Gemma3 may not have one)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading Gemma3 model with 4-bit quantization: {self.model_name}")

        # QLoRA requires 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Use Gemma3ForCausalLM specifically
        self.base_model = Gemma3ForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,  # Important: use float16 with QLoRA
            attn_implementation="eager",  # Gemma3 optimization
        )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable()

        # Configure LoRA for Gemma3
        target_modules = self._get_target_modules()
        logger.info(f"Applying LoRA to modules: {target_modules}")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        self.base_model = prepare_model_for_kbit_training(self.base_model)
        # Apply LoRA to model
        self.model = get_peft_model(self.base_model, lora_config)

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

        train_dataset = QLoRADataset(train_examples, self.tokenizer, max_length)

        eval_dataset = None
        if len(eval_examples) > 0:
            eval_dataset = QLoRADataset(eval_examples, self.tokenizer, max_length)

        return train_dataset, eval_dataset

    def train(
        self,
        val_split=0.1,
        num_epochs=3,
        learning_rate=2e-4,
        batch_size=1,  # Lower for Gemma3
        gradient_accumulation_steps=8,  # Higher to compensate
        max_length=2500,
        warmup_ratio=0.1,  # Higher warmup for Gemma3
        weight_decay=0.01,
        save_steps=500,
        eval_steps=500,
        logging_steps=10,
    ):
        """Train the model with QLoRA"""
        logger.info("Starting QLoRA training for Gemma3")

        # Prepare datasets
        train_dataset, eval_dataset = self._prepare_datasets(
            val_split=val_split,
            max_length=max_length,
        )

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, pad_to_multiple_of=8
        )

        # Training arguments optimized for Gemma3 + QLoRA
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
            logging_steps=logging_steps,
            save_strategy="steps",
            save_steps=save_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=eval_steps if eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            remove_unused_columns=False,
            seed=self.seed,
            fp16=True,  # Essential for QLoRA
            bf16=False,  # Use fp16, not bf16 for QLoRA
            dataloader_pin_memory=False,
            group_by_length=True,
            report_to=None,  # Disable wandb/tensorboard
            gradient_checkpointing=True,
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

        logger.info("QLoRA training complete!")
        return self.model, trainer

    def generate_summary(
        self, text, max_new_tokens=512, temperature=0.1, do_sample=False
    ):
        """Generate a summary using the QLoRA-trained Gemma3 model"""
        messages = [
            {
                "role": "user",
                "content": f"{system_prompt}\n\nStreść poniższy dokument:\n{text}",
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(self.model.device)

        input_len = inputs.shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Decode only the generated part
        output_text = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )

        return output_text.strip()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="QLoRA Fine-tuning Pipeline for Gemma3 Models"
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-4b-it",
        help="Name or path of the Gemma3 model to fine-tune",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/qlora_gemma3",
        help="Directory to save the trained model and outputs",
    )

    # LoRA parameters
    parser.add_argument(
        "--lora_r", type=int, default=16, help="LoRA rank (higher = more parameters)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (usually 2x rank)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout for regularization",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate for training"
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
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduler",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization",
    )

    # Evaluation and saving
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save model every N steps"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="Evaluate model every N steps"
    )

    parser.add_argument(
        "--max_length", type=int, default=2000, help="Maximum sequence length"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(
        "Starting QLoRA Fine-tuning Pipeline for Gemma3 with the following configuration:"
    )
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"LoRA rank: {args.lora_r}")
    logger.info(f"LoRA alpha: {args.lora_alpha}")
    logger.info(f"LoRA dropout: {args.lora_dropout}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Batch size: {args.batch_size}")

    # Initialize QLoRA pipeline for Gemma3
    pipeline = QLoRAPipeline(
        model_name=args.model_name,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Train the model
    model, trainer = pipeline.train(
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        val_split=args.val_split,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )


if __name__ == "__main__":
    main()
