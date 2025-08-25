import torch
from loguru import logger
import numpy as np
from pathlib import Path

# from datasets import Dataset
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    Gemma3ForCausalLM,
    BitsAndBytesConfig,
)

from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
from data_preprocess import system_prompt, create_train_data_for_prompt_tuning

DOCUMENTS_PATH = "data/processed/documents/"
SUMMARIES_PATH = "data/processed/summaries/done/"


class LegalSummaryDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length, system_prompt):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example["input"]
        target_text = example["target"]

        # Create chat template
        messages = [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": target_text},
        ]
        # messages = [
        #     {"role": "user", "content": f"{self.system_prompt}\n\n{input_text}"},
        #     {"role": "assistant", "content": target_text},
        # ]
        # messages = [
        #     {
        #         "role": "system",
        #         "content": [
        #             {
        #                 "type": "text",
        #                 "text": self.system_prompt,
        #             }
        #         ],
        #     },
        #     {"role": "user", "content": [{"type": "text", "text": prompt}]},
        # ]

        # Apply chat template
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False,
        )
        user_messages = [{"role": "user", "content": f"{input_text}"}]

        # Get tokenized sequence up to (and including) assistant marker
        user_tokens = self.tokenizer.apply_chat_template(
            user_messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False,
        ).squeeze(0)

        # Remove batch dimension
        tokenized = tokenized.squeeze(0)

        logger.info(tokenized.shape)
        # Truncate if needed
        if len(tokenized) > self.max_length:
            tokenized = tokenized[: self.max_length]

        logger.info(tokenized.shape)
        # Create attention mask
        attention_mask = torch.ones_like(tokenized)

        # Create labels (copy of input_ids)
        labels = tokenized.clone()

        user_messages = [
            {"role": "user", "content": f"{self.system_prompt}\n\n{input_text}"}
        ]

        # Get tokenized sequence with assistant marker
        user_tokens = self.tokenizer.apply_chat_template(
            user_messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,  # Include the assistant marker
        ).squeeze(0)

        # If user_tokens length is less than full sequence, mask everything up to that point
        if len(user_tokens) < len(tokenized):
            labels[: len(user_tokens)] = -100

        logger.info(tokenized.shape)
        return {
            "input_ids": tokenized,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class PromptTuningPipeline:
    def __init__(
        self,
        model_name="google/gemma-3-4b-it",
        output_dir="./output/prompt_tuning",
        num_virtual_tokens=20,
        prompt_tuning_init_text="Streść dokument sądowy zgodnie z wymogami:",
        torch_dtype="bfloat16",
        seed=42,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_virtual_tokens = num_virtual_tokens
        self.prompt_tuning_init_text = prompt_tuning_init_text
        self.torch_dtype = (
            torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16
        )
        self.seed = seed

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

        logger.info(f"Loading model: {self.model_name}")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        self.base_model = Gemma3ForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.base_model.gradient_checkpointing_enable()

        # Configure prompt tuning
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=self.num_virtual_tokens,
            prompt_tuning_init_text=self.prompt_tuning_init_text,
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

    def _prepare_datasets(
        self,
        val_split=0.1,
        max_source_length=4096,
        max_target_length=512,
    ):
        """Prepare training and validation datasets"""
        max_length = max_source_length + max_target_length

        logger.info("loading training data from documents and summaries")
        train_data = create_train_data_for_prompt_tuning(
            documents_path=DOCUMENTS_PATH, target_path=SUMMARIES_PATH, max_len=5000
        )
        logger.info("Loaded train data")

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

        train_dataset = LegalSummaryDataset(
            train_examples, self.tokenizer, max_length, system_prompt
        )

        eval_dataset = None
        if len(eval_examples) > 0:
            eval_dataset = LegalSummaryDataset(
                eval_examples, self.tokenizer, max_length, system_prompt
            )

        return train_dataset, eval_dataset

    def train(
        self,
        val_split=0.1,
        num_epochs=6,
        learning_rate=0.035,
        batch_size=1,
        gradient_accumulation_steps=4,
        max_source_length=2500,
        max_target_length=512,
    ):
        """Train the model with prompt tuning"""
        logger.info("Starting prompt tuning training")

        # Prepare datasets
        train_dataset, eval_dataset = self._prepare_datasets(
            val_split=val_split,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )

        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset else "no",
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset else False,
            remove_unused_columns=False,
            seed=self.seed,
            fp16=self.torch_dtype == torch.float16,
            bf16=self.torch_dtype == torch.bfloat16,
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

    def generate_summary(self, text, max_new_tokens=512):
        """Generate a summary using the trained model"""
        prompt = f"Streść poniższy dokument:\n{text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(self.model.device)

        input_len = inputs.shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        output_text = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        return output_text


def main():
    # Example usage
    pipeline = PromptTuningPipeline(
        model_name="google/gemma-3-4b-it",
        output_dir="./output/prompt_tuning",
        num_virtual_tokens=20,
    )

    # Train the model
    pipeline.train(
        num_epochs=50, learning_rate=0.0035, batch_size=1, gradient_accumulation_steps=8
    )


if __name__ == "__main__":
    main()
