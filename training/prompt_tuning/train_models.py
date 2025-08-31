import argparse
from loguru import logger

from peft import PromptTuningConfig, TaskType
from training.utils import BasePEFTTrainer


class PromptTuningTrainer(BasePEFTTrainer):

    def __init__(self, model_name: str, num_virtual_tokens: int = 20):
        super().__init__(model_name)
        self.num_virtual_tokens = num_virtual_tokens

    def _create_peft_config(self, **kwargs):
        logger.info(
            f"Setting up prompt tuning with {self.num_virtual_tokens} virtual tokens"
        )

        return PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=self.num_virtual_tokens,
            tokenizer_name_or_path=self.model_name,
        )

    def _prepare_model_for_peft(self):
        return self.preparer.model

    def setup_model_and_prompt_tuning(self, **kwargs):
        return self.setup_model_and_peft(**kwargs)

    def _save_model_and_info(
        self,
        output_dir,
        train_result,
        training_config,
        train_dataset_size,
    ):
        prompt_tuning_config = {
            "num_virtual_tokens": self.num_virtual_tokens,
        }

        return super()._save_model_and_info(
            output_dir=output_dir,
            train_result=train_result,
            training_config=training_config,
            train_dataset_size=train_dataset_size,
            peft_config=prompt_tuning_config,
        )


def train_prompt_tuning_model(
    model_name: str,
    output_dir: str,
    num_virtual_tokens: int = 20,
    epochs: int = 6,
    learning_rate: float = 0.3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_length: int = 3000,
    seed: int = 42,
) -> str:
    """Train a prompt tuning model using preparers"""

    trainer = PromptTuningTrainer(
        model_name=model_name,
        num_virtual_tokens=num_virtual_tokens,
    )
    trainer.setup_model_and_prompt_tuning()

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
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
