import os
import json
import torch
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Gemma3ForCausalLM,
)
from peft import PeftModel
from data_preprocess import system_prompt, get_doc_text
from metrics import calculate_metrics
import nltk
import gc

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("wordnet")
except LookupError:
    nltk.download("punkt")
    nltk.download("wordnet")

# Updated paths for train/test split
TRAIN_DOCUMENTS_PATH = "data/training/train/documents"
TRAIN_SUMMARIES_PATH = "data/training/train/summaries"
TEST_DOCUMENTS_PATH = "data/training/test/documents"
TEST_SUMMARIES_PATH = "data/training/test/summaries"
RESULTS_DIR = "evaluation_results/"


class TimingContext:
    """Context manager for timing operations"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time


class BaseModelEvaluator(ABC):
    """Base abstract class for model evaluation on test set"""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        config_name: str,
        test_documents_dir: str = TEST_DOCUMENTS_PATH,
        test_summaries_dir: str = TEST_SUMMARIES_PATH,
        results_dir: str = RESULTS_DIR,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.config_name = config_name
        self.test_documents_dir = test_documents_dir
        self.test_summaries_dir = test_summaries_dir
        self.results_dir = Path(results_dir) / model_name / config_name
        self.model = None
        self.tokenizer = None

        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer - to be implemented by subclasses"""
        pass

    @abstractmethod
    def generate_summary(self, document_path: str) -> str:
        """Generate summary for a document - to be implemented by subclasses"""
        pass

    def unload_model(self):
        """Unload model and tokenizer to free up memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Unloaded model: {self.config_name}")

    def _calculate_tokens_per_second(self, text: str, inference_time: float) -> float:
        """Estimate tokens per second (rough approximation)"""
        if inference_time == 0:
            return 0
        # Rough approximation: 1 token ≈ 0.75 words
        word_count = len(text.split())
        estimated_tokens = word_count / 0.75
        return estimated_tokens / inference_time

    def evaluate_on_test_set(self):
        """Evaluate model on test set only"""
        print(f"Loading model: {self.config_name}")
        with TimingContext() as model_loading_timer:
            self._load_model_and_tokenizer()
        print(f"Model loaded in {model_loading_timer.duration:.2f} seconds")

        results = []
        inference_times = []

        # Get test document files
        test_files = [
            f for f in os.listdir(self.test_documents_dir) if f.endswith(".txt")
        ]

        # show progrss bar
        for doc_file in tqdm(test_files, desc=f"Evaluating {self.config_name}"):
            doc_path = os.path.join(self.test_documents_dir, doc_file)
            reference_path = os.path.join(self.test_summaries_dir, doc_file)

            # Skip if reference summary doesn't exist
            if not os.path.exists(reference_path):
                print(f"Skipping {doc_file} - no reference summary found")
                continue

            # Generate summary with timing
            with TimingContext() as inference_timer:
                generated_summary = self.generate_summary(doc_path)

            inference_time = inference_timer.duration
            inference_times.append(inference_time)

            # Get reference summary
            reference_summary = get_doc_text(reference_path)

            # Calculate metrics
            metrics = calculate_metrics(reference_summary, generated_summary)

            # Add results with timing information
            results.append(
                {
                    "filename": doc_file,
                    "generated_summary": generated_summary,
                    "reference_summary": reference_summary,
                    "inference_time_seconds": inference_time,
                    "tokens_per_second": self._calculate_tokens_per_second(
                        generated_summary, inference_time
                    ),
                    **metrics,
                }
            )

        # Unload model
        self.unload_model()

        return results, inference_times

    def run_evaluation(self):
        """Run evaluation and save results"""
        results, inference_times = self.evaluate_on_test_set()

        # Calculate average metrics
        if results:
            avg_metrics = {
                "config_name": self.config_name,
                "model_name": self.model_name,
                "model_path": self.model_path,
                "avg_rougeL": sum(r["rouge-l"] for r in results) / len(results),
                "avg_bleu": sum(r["bleu"] for r in results) / len(results),
                "avg_meteor": sum(r["meteor"] for r in results) / len(results),
                "avg_inference_time_seconds": sum(inference_times)
                / len(inference_times),
                "min_inference_time_seconds": min(inference_times),
                "max_inference_time_seconds": max(inference_times),
                "total_inference_time_seconds": sum(inference_times),
                "avg_tokens_per_second": sum(r["tokens_per_second"] for r in results)
                / len(results),
                "test_document_count": len(results),
            }

            # Save detailed results
            output_file = self.results_dir / "test_evaluation_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "config_name": self.config_name,
                        "model_info": {
                            "model_name": self.model_name,
                            "model_path": self.model_path,
                        },
                        "summary_metrics": avg_metrics,
                        "detailed_results": results,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"Test evaluation complete. Processed {len(results)} documents.")
            print(f"Results saved to {output_file}")
            return avg_metrics, results
        else:
            print("No test documents were evaluated. Check your data paths.")
            return None, []


class StandardModelEvaluator(BaseModelEvaluator):
    """Evaluator for Qwen, Llama, and other standard chat models"""

    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with quantization and LoRA adapter"""
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        print(f"Loading base model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Load LoRA adapter if model_path points to adapter
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading LoRA adapter from {self.model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)

            # Verify adapter loading
            if hasattr(self.model, "get_nb_trainable_parameters"):
                trainable, total = self.model.get_nb_trainable_parameters()
                print(
                    f"LoRA adapter loaded: {trainable:,} trainable parameters ({100*trainable/total:.3f}%)"
                )

        self.model.eval()

    def generate_summary(self, document_path: str) -> str:
        """Generate summary using standard chat format"""
        document_text = get_doc_text(path=document_path)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Streść poniższy dokument:\n{document_text}"},
        ]

        try:
            if "Qwen3" in self.model_name:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception as e:
            print(f"Chat template failed, using fallback format: {e}")
            text = f"System: {system_prompt}\n\nUser: Streść poniższy dokument:\n{document_text}\n\nAssistant: "

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                temperature=0.1,
            )

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ][0]

        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()


class GemmaModelEvaluator(BaseModelEvaluator):
    """Evaluator for Gemma models"""

    def _load_model_and_tokenizer(self):
        """Load Gemma model and tokenizer with quantization and LoRA adapter"""
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        print(f"Loading base Gemma model: {self.model_name}")
        self.model = Gemma3ForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager",  # More stable for Gemma
        )

        # Load LoRA adapter if model_path points to adapter
        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading LoRA adapter from {self.model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)

            # Verify adapter loading
            if hasattr(self.model, "get_nb_trainable_parameters"):
                trainable, total = self.model.get_nb_trainable_parameters()
                print(
                    f"LoRA adapter loaded: {trainable:,} trainable parameters ({100*trainable/total:.3f}%)"
                )

        self.model.eval()

    def generate_summary(self, document_path: str) -> str:
        """Generate summary using Gemma chat format (no system role)"""
        document_text = get_doc_text(path=document_path)

        messages = [
            {
                "role": "user",
                "content": f"{system_prompt}\n\nStreść poniższy dokument:\n{document_text}",
            },
        ]

        try:
            if "qwen3" in self.model_name:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception as e:
            print(f"Chat template failed, using fallback format: {e}")
            text = f"User: {system_prompt}\n\nStreść poniższy dokument:\n{document_text}\n\nAssistant: "

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                temperature=0.1,
            )

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ][0]

        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()


def create_evaluator(
    model_name: str, model_path: str, config_name: str, **kwargs
) -> BaseModelEvaluator:
    """Factory function to create appropriate evaluator based on model name"""
    if "gemma" in model_name.lower():
        return GemmaModelEvaluator(model_name, model_path, config_name, **kwargs)
    else:
        return StandardModelEvaluator(model_name, model_path, config_name, **kwargs)


def evaluate_trained_models(training_output_dir: str, model_name: str):
    """
    Evaluate all trained models from the training pipeline output directory

    Args:
        training_output_dir: Directory containing trained model configs (e.g., "./output/multi_config_training")
        model_name: Base model name used for training (e.g., "Qwen/Qwen3-4B")
    """
    training_output_path = Path(training_output_dir)

    if not training_output_path.exists():
        print(f"Training output directory not found: {training_output_dir}")
        return

    # Find all config directories (should contain adapter files)
    config_dirs = [
        d
        for d in training_output_path.iterdir()
        if d.is_dir() and d.name != "__pycache__"
    ]

    if not config_dirs:
        print(f"No trained model configs found in {training_output_dir}")
        return

    print(f"Found {len(config_dirs)} trained configurations to evaluate:")
    for config_dir in config_dirs:
        print(f"  - {config_dir.name}")

    all_results = []

    # Evaluate each trained configuration
    for config_dir in config_dirs:
        config_name = config_dir.name

        # Check if this directory contains a trained model
        if not (config_dir / "adapter_config.json").exists():
            print(f"Skipping {config_name} - no adapter_config.json found")
            continue

        print(f"\n{'='*60}")
        print(f"EVALUATING: {config_name}")
        print(f"{'='*60}")

        # Create evaluator for this config
        evaluator = create_evaluator(
            model_name=model_name,
            model_path=str(config_dir),
            config_name=config_name,
            test_documents_dir=TEST_DOCUMENTS_PATH,
            test_summaries_dir=TEST_SUMMARIES_PATH,
            results_dir=RESULTS_DIR,
        )

        # Run evaluation
        metrics, _ = evaluator.run_evaluation()
        if metrics:
            all_results.append(metrics)

    # Create comparison report
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Save comparison to CSV
        comparison_path = Path(RESULTS_DIR) / "test_set_comparison.csv"
        results_df.to_csv(comparison_path, index=False)
        print(f"Detailed comparison saved to: {comparison_path}")

        return all_results

    else:
        print("❌ No successful evaluations completed.")
        return []


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained models on test set")
    parser.add_argument(
        "--training_output_dir",
        type=str,
        required=True,
        help="Directory containing trained model configurations",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Base model name used for training",
    )

    args = parser.parse_args()

    # Run evaluation
    evaluate_trained_models(args.training_output_dir, args.model_name)


if __name__ == "__main__":
    main()
