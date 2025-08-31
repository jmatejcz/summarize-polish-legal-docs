import argparse
from loguru import logger

from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from training.utils import BasePEFTTrainer


class QLoRATrainer(BasePEFTTrainer):

    def __init__(self, model_name: str):
        super().__init__(model_name)

    def _get_target_modules(self) -> list:
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

    def _create_peft_config(
        self, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1, **kwargs
    ):
        """Create LoRA configuration"""
        logger.info(
            f"Setting up LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}"
        )

        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=self._get_target_modules(),
            bias="none",
        )

    def _prepare_model_for_peft(self):
        return prepare_model_for_kbit_training(self.preparer.model)

    def setup_model_and_lora(
        self, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1
    ):
        return self.setup_model_and_peft(
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )

    def _create_training_args(self, **kwargs):
        # Add QLoRA-specific defaults
        qlora_defaults = {
            "gradient_checkpointing": True,
            "max_grad_norm": 0.3,
        }
        qlora_defaults.update(kwargs)

        return super()._create_training_args(**qlora_defaults)

    def _save_model_and_info(
        self,
        output_dir,
        train_result,
        training_config,
        train_dataset_size,
        peft_config=None,
    ):
        if peft_config is None and hasattr(self.model, "peft_config"):
            peft_module = list(self.model.peft_config.keys())[0]
            lora_config = self.model.peft_config[peft_module]
            peft_config = {
                "lora_r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "lora_dropout": lora_config.lora_dropout,
                "target_modules": lora_config.target_modules,
            }

        return super()._save_model_and_info(
            output_dir=output_dir,
            train_result=train_result,
            training_config=training_config,
            train_dataset_size=train_dataset_size,
            peft_config=peft_config,
        )


def train_qlora_model(
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
) -> str:
    """Train a QLoRA model using preparers"""

    trainer = QLoRATrainer(model_name)
    trainer.setup_model_and_lora(
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
    )

    trainer.train(
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        seed=seed,
    )

    return output_dir


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

    logger.info("Starting QLoRA Training Pipeline with the following configuration:")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(
        f"LoRA r: {args.lora_r}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}"
    )
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")

    train_qlora_model(
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
