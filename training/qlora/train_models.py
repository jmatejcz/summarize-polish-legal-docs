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
    Gemma3ForCausalLM,
    BitsAndBytesConfig,
)

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from data_preprocess import (
    create_train_data_for_prompt_tuning,
)
from evaluate_models import SYSTEM_PROMPT


# Updated paths for train/test split
TRAIN_DOCUMENTS_PATH = "data/training/train/documents"
TRAIN_SUMMARIES_PATH = "data/training/train/summaries"


class QLoRADataset(Dataset):
    """Dataset class for QLoRA training"""

    def __init__(self, examples, tokenizer, max_length=4096, is_gemma=False):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_gemma = is_gemma

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example["input"]
        target_text = example["target"]

        # Format based on model type
        if self.is_gemma:
            # Gemma format (no system role)
            messages = [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nStreść poniższy dokument:\n{input_text}",
                },
                {"role": "assistant", "content": target_text},
            ]

            # Tokenize directly for Gemma
            full_tokens = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
            ).squeeze(0)

            prompt_messages = [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nStreść poniższy dokument:\n{input_text}",
                }
            ]

            prompt_tokens = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).squeeze(0)

            # Truncate if needed
            if len(full_tokens) > self.max_length:
                full_tokens = full_tokens[: self.max_length]

            attention_mask = torch.ones_like(full_tokens)
            labels = full_tokens.clone()

            # Mask prompt tokens
            if len(prompt_tokens) < len(full_tokens):
                labels[: len(prompt_tokens)] = -100

            return {
                "input_ids": full_tokens,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        else:
            # Standard format (with system role)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Streść poniższy dokument:\n{input_text}"},
                {"role": "assistant", "content": target_text},
            ]
            # Two-step tokenization for standard models
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Streść poniższy dokument:\n{input_text}"},
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize both
            full_tokens = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt",
            )
            prompt_tokens = self.tokenizer(
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

            if prompt_len < len(labels):
                labels[:prompt_len] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }


def load_model_and_tokenizer(model_name, lora_r, lora_alpha, lora_dropout):
    """Load model and tokenizer with LoRA configuration"""
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    logger.info(f"Loading base model: {model_name}")

    # Load appropriate model class
    model_args = {
        "quantization_config": quantization_config,
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }

    if "gemma" in model_name.lower():
        model_args["attn_implementation"] = "eager"
        base_model = Gemma3ForCausalLM.from_pretrained(model_name, **model_args)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)

    # Enable gradient checkpointing
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    # Get target modules based on model
    if any(name in model_name.lower() for name in ["qwen", "gemma", "llama", "bielik"]):
        target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    else:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    # Apply LoRA
    base_model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(base_model, lora_config)

    # Log parameters
    trainable_params, all_params = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)"
    )

    return model, tokenizer


def train_model(
    model_name,
    output_dir,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    epochs=3,
    learning_rate=2e-4,
    batch_size=1,
    gradient_accumulation_steps=4,
    max_length=3072,
    seed=42,
):
    """Train a single QLoRA model"""

    torch.manual_seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training with config:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  LoRA r={lora_r}, α={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"  Epochs: {epochs}, LR: {learning_rate}")
    logger.info(f"  Output: {output_dir}")

    model, tokenizer = load_model_and_tokenizer(
        model_name, lora_r, lora_alpha, lora_dropout
    )

    logger.info("Loading training data...")
    train_data = create_train_data_for_prompt_tuning(
        documents_path=TRAIN_DOCUMENTS_PATH,
        target_path=TRAIN_SUMMARIES_PATH,
        max_len=max_length,
    )
    logger.info(f"Loaded {len(train_data)} training examples")

    is_gemma = "gemma" in model_name.lower()
    train_dataset = QLoRADataset(train_data, tokenizer, max_length, is_gemma)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
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
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    logger.info("Saving model...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    training_info = {
        "model_name": model_name,
        "lora_config": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
        },
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


def main():
    parser = argparse.ArgumentParser(description="Simple QLoRA Training")

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
