import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Gemma3ForCausalLM,
)
from peft import PeftModel, PeftConfig
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

PROCESSED_DIR = "data/processed/documents/"
SUMMARIES_DIR = "data/processed/summaries/done"
RESULTS_DIR = "results/"


class BaseSummarizationEvaluator(ABC):
    """Base abstract class for summarization evaluation"""

    def __init__(
        self,
        model_name,
        processed_dir="data/processed/documents/",
        summaries_dir="data/summaries/",
        results_dir="data/results/",
        dir_name: str = "qwen3_fixed",
        max_docs=None,
    ):
        self.model_name = model_name
        self.processed_dir = processed_dir
        self.summaries_dir = summaries_dir
        self.results_dir = Path(results_dir) / dir_name
        self.max_docs = max_docs
        self.model = None
        self.tokenizer = None

        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer - to be implemented by subclasses"""
        pass

    @abstractmethod
    def generate_summary(self, document_path):
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
        # Force garbage collection
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Unloaded model: {self.model_name}")

    def evaluate_all_documents(self):
        """Evaluate all documents with available reference summaries"""
        # Load model and tokenizer
        print(f"Loading model: {self.model_name}")
        self._load_model_and_tokenizer()
        print("Model loaded successfully")

        results = []

        # Get list of all document files
        document_files = [
            f for f in os.listdir(self.processed_dir) if f.endswith(".txt")
        ]

        # Limit number of documents if specified
        if self.max_docs:
            document_files = document_files[: self.max_docs]

        print(f"Found {len(document_files)} documents to evaluate")

        for doc_file in tqdm(document_files, desc="Evaluating documents"):
            doc_path = os.path.join(self.processed_dir, doc_file)
            reference_path = os.path.join(self.summaries_dir, doc_file)

            # Skip if reference summary doesn't exist
            if not os.path.exists(reference_path):
                print(f"Skipping {doc_file} - no reference summary found")
                continue

            # Generate summary
            generated_summary = self.generate_summary(doc_path)

            # Get reference summary
            reference_summary = get_doc_text(reference_path)

            # Calculate metrics
            metrics = calculate_metrics(reference_summary, generated_summary)

            # Add results
            results.append(
                {
                    "filename": doc_file,
                    "generated_summary": generated_summary,
                    "reference_summary": reference_summary,
                    **metrics,
                }
            )

        # Always unload the model, even if an error occurs
        self.unload_model()

        return results

    def run_evaluation(self):
        """Run the complete evaluation pipeline and save results"""
        print(f"Starting evaluation pipeline for {self.model_name}...")

        # Evaluate all documents
        results = self.evaluate_all_documents()

        # Calculate average metrics
        if results:
            avg_metrics = {
                "model": self.model_name,
                "avg_rougeL": sum(r["rouge-l"] for r in results) / len(results),
                "avg_bleu": sum(r["bleu"] for r in results) / len(results),
                "avg_meteor": sum(r["meteor"] for r in results) / len(results),
                # "avg_bertscore": sum(r["bertscore"] for r in results) / len(results),
                "document_count": len(results),
            }

            # Save detailed results
            output_file = self.results_dir / "evaluation_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model": self.model_name,
                        "summary": avg_metrics,
                        "details": results,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"Evaluation complete. Processed {len(results)} documents.")
            print("Average metrics:")
            for k, v in avg_metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

            return avg_metrics, results
        else:
            print("No documents were evaluated. Check your data paths.")
            return None, []


class GemmaSummarizationEvaluator(BaseSummarizationEvaluator):
    """Gemma-specific implementation of summarization evaluation"""

    def __init__(
        self,
        model_name,
        processed_dir="data/processed/documents/",
        summaries_dir="data/summaries/",
        results_dir="data/results/",
        max_docs=None,
        peft_path=None,
        quantize=False,
    ):
        self.peft_path = peft_path
        self.quantize = quantize
        super().__init__(
            model_name, processed_dir, summaries_dir, results_dir, max_docs
        )

    def _load_model_and_tokenizer(self):
        """Load Gemma model and tokenizer with optional PEFT adapter"""
        if self.peft_path:
            # Load with PEFT adapter (prompt tuning)
            print(f"Loading PEFT model from {self.peft_path}")
            self.model, self.tokenizer = self._load_model_with_peft()
        else:
            # Load base model only
            print(f"Loading base Gemma model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if self.quantize:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
                self.model = Gemma3ForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                ).eval()
            else:
                self.model = Gemma3ForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                ).eval()

    def _load_model_with_peft(self):
        """Load the prompt-tuned model using PEFT"""
        # Load the adapter configuration
        peft_config = PeftConfig.from_pretrained(self.peft_path)

        # Load base model with same configuration as training
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

        base_model = Gemma3ForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,  # This will be "google/gemma-3-4b-it"
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map={"": 0},
            attn_implementation="flash_attention_2",
        )

        # Load the PEFT model (this loads the adapter_model.safetensors)
        model = PeftModel.from_pretrained(base_model, self.peft_path)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return model, tokenizer

    def generate_summary(self, document_path):
        """Generate summary using Gemma model"""
        document_text = get_doc_text(path=document_path)
        prompt = f"Streść poniższy dokument:\n{document_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        # inputs = self.tokenizer.apply_chat_template(
        #     messages,
        #     add_generation_prompt=True,
        #     tokenize=True,
        #     return_tensors="pt",
        # ).to('cuda')

        input_len = inputs["input_ids"].shape[-1]
        # print(inputs.shape)
        # attention_mask = torch.ones_like(inputs)
        with torch.no_grad():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generation = generation[0][input_len:]

        decoded = self.tokenizer.decode(generation, skip_special_tokens=True)
        return decoded.strip()


class QwenSummarizationEvaluator(BaseSummarizationEvaluator):
    """Qwen-specific implementation of summarization evaluation"""

    def __init__(
        self,
        model_name,
        dir_name: str,
        processed_dir="data/processed/documents/",
        summaries_dir="data/summaries/",
        results_dir="data/results/",
        max_docs=None,
        quantize=False,
        quantize_bits=4,
        prompt_tuning_path=None,
        enable_thinking=False,
    ):
        self.quantize = quantize
        self.quantize_bits = quantize_bits
        self.prompt_tuning_path = prompt_tuning_path
        self.enable_thinking = enable_thinking
        super().__init__(
            model_name, processed_dir, summaries_dir, results_dir, dir_name, max_docs
        )

    def _load_model_and_tokenizer(self):
        """Load Qwen model and tokenizer with quantization and prompt tuning options"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Configure quantization if needed
        quantization_config = None
        if self.quantize:
            if self.quantize_bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
            elif self.quantize_bits == 8:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load base model
        if self.quantize:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        # Apply prompt tuning if specified
        if self.prompt_tuning_path:
            print(f"Loading prompt tuning from {self.prompt_tuning_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.prompt_tuning_path,
            )
            # Set to evaluation mode
            self.model.eval()

    def generate_summary(self, document_path):
        """Generate summary using Qwen model"""
        document_text = get_doc_text(path=document_path)
        prompt = f"Streść poniższy dokument:\n{document_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ][0]

        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()


def evaluate_models():
    """Run evaluation for all configured models sequentially"""
    # Paths configuration
    max_docs = 60
    evaluators = [
        # Bielik models
        QwenSummarizationEvaluator(
            model_name="speakleash/Bielik-4.5B-v3.0-Instruct",
            processed_dir=PROCESSED_DIR,
            summaries_dir=SUMMARIES_DIR,
            results_dir=RESULTS_DIR,
            dir_name="bielik_base",
            quantize=True,
            max_docs=max_docs,
        ),
        QwenSummarizationEvaluator(
            model_name="speakleash/Bielik-4.5B-v3.0-Instruct",
            processed_dir=PROCESSED_DIR,
            summaries_dir=SUMMARIES_DIR,
            results_dir=RESULTS_DIR,
            dir_name="bielik-4.5b-conservative",
            prompt_tuning_path="./summarize/prompt_tuning/results/models/bielik-4.5b-conservative",
            quantize=True,
            max_docs=max_docs,
        ),
        QwenSummarizationEvaluator(
            model_name="speakleash/Bielik-4.5B-v3.0-Instruct",
            processed_dir=PROCESSED_DIR,
            summaries_dir=SUMMARIES_DIR,
            results_dir=RESULTS_DIR,
            dir_name="bielik-4.5b-moderate",
            prompt_tuning_path="./summarize/prompt_tuning/results/models/bielik-4.5b-moderate",
            quantize=True,
            max_docs=max_docs,
        ),
        # Qwen3 models
        QwenSummarizationEvaluator(
            model_name="Qwen/Qwen3-4B",
            processed_dir=PROCESSED_DIR,
            summaries_dir=SUMMARIES_DIR,
            results_dir=RESULTS_DIR,
            dir_name="qwen3_base",
            quantize=True,
            max_docs=max_docs,
        ),
        QwenSummarizationEvaluator(
            model_name="Qwen/Qwen3-4B",
            processed_dir=PROCESSED_DIR,
            summaries_dir=SUMMARIES_DIR,
            results_dir=RESULTS_DIR,
            dir_name="qwen3-4b-conservative",
            prompt_tuning_path="./summarize/prompt_tuning/results/models/qwen3-4b-conservative",
            quantize=True,
            max_docs=max_docs,
        ),
        QwenSummarizationEvaluator(
            model_name="Qwen/Qwen3-4B",
            processed_dir=PROCESSED_DIR,
            summaries_dir=SUMMARIES_DIR,
            results_dir=RESULTS_DIR,
            dir_name="qwen3-4b-moderate",
            prompt_tuning_path="./summarize/prompt_tuning/results/models/qwen3-4b-moderate",
            quantize=True,
            max_docs=max_docs,
        ),
    ]

    # Collect all results
    all_results = []

    # Run evaluation for each model sequentially
    for i, evaluator in enumerate(evaluators):
        print(f"\n{'='*50}")
        print(f"Evaluating model {i+1}/{len(evaluators)}: {evaluator.model_name}")
        if hasattr(evaluator, "peft_path") and evaluator.peft_path:
            print(f"With PEFT adapter: {evaluator.peft_path}")
        print(f"{'='*50}")

        metrics, _ = evaluator.run_evaluation()
        if metrics:
            all_results.append(metrics)

    # Compare all models
    if all_results:
        # Create comparison table
        results_df = pd.DataFrame(all_results)

        # Save comparison to CSV
        comparison_path = Path(RESULTS_DIR) / "model_comparison.csv"
        results_df.to_csv(comparison_path, index=False)

        print(f"\nModel comparison saved to {comparison_path}")
        print("\nModel Comparison:")
        print(results_df.to_string())

        # Show which model performed best
        best_rouge = results_df.loc[results_df["avg_rougeL"].idxmax()]
        best_bleu = results_df.loc[results_df["avg_bleu"].idxmax()]

        print(f"\nBest ROUGE-L: {best_rouge['model']} ({best_rouge['avg_rougeL']:.4f})")
        print(f"Best BLEU: {best_bleu['model']} ({best_bleu['avg_bleu']:.4f})")
    else:
        print("No successful evaluations completed.")


if __name__ == "__main__":
    # For full evaluation, run all models
    evaluate_models()
