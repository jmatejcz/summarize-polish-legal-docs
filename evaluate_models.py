import os
import json
import torch
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union, Literal
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Gemma3ForCausalLM,
)
from peft import PeftModel, PeftConfig
from data_preprocess import get_doc_text
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

SYSTEM_PROMPT = """Jesteś asystentem wyspecjalizowanym w streszczaniu polskich dokumentów sądowych. 
W zależności od rodzaju pisma (pozew lub odpowiedź na pozew) twoje streszczenia MUSZĄ zawierać następujące elementy:

1. Pozew:
   - Wartość przedmiotu sporu (dokładna kwota, o którą powód się zwraca)
   - Roszczenia powoda (o co powód wnosi, jakie podnosi zarzuty)
   - Wnioski dowodowe:
     - Świadkowie (imię i nazwisko - jeżeli pojawią się w dokumencie)
     - Biegli (zakres specjalizacji i imię i nazwisko biegłego - jeżeli pojawią się w dokumencie)
     - Inne dowody (dokumenty, akta, ekspertyzy, strony internetowe itp. - jeżeli pojawią się w dokumencie)

2. Odpowiedź na pozew:
    - Stanowisko pozwanego (czy wnosi o oddalenie powództwa w całości/części, czy uznaje roszczenia)
    - Wnioski dowodowe:
     - Świadkowie (imię i nazwisko - jeżeli pojawią się w dokumencie)
     - Biegli (zakres specjalizacji i imię i nazwisko - jeżeli pojawią się w dokumencie)
     - Inne dowody (dokumenty, akta, ekspertyzy, internetowe itp. - jeżeli pojawią się w dokumencie)

Zasady formatowania:
- Każdy punkt powinien być krótki, zwięzły i punktowany.
- Wyodrębnij wyłącznie te informacje, które są faktycznie zawarte w dokumencie.
- Nie dodawaj własnych interpretacji ani komentarzy – skup się na suchych faktach.
- Jeżeli jest jasno opisane, dodaj do każdego dowodu w jakim celu/ na jaki fakt jest on podnoszony.

===  PRZYKŁADY DLA POZWU === 

1.  Wartość przedmiotu sporu: 2511,16 zł
2.  Roszczenia powoda:
        - Zasądzenie od pozwanego na rzecz powoda kwoty 2511,16 zł tytułem odszkodowania wraz z odsetkami za opóźnienie od 11.10.2021 r.
3.  Wnioski dowodowe:
        - Akta szkody nr 4897144/1.
        - Faktura VAT nr FV/VB/127/23/I.
        - Umowa cesji wraz z pełnomocnictwem.
        - Opinia biegłego z zakresu wyceny pojazdów mechanicznych
        
===============================================
1.  Wartość przedmiotu sporu: 1981,80 zł
2.  Roszczenia powoda:
    - zasądzenie od pozwanego kwoty 1981,80 zł tytułem odszkodowania wraz z odsetkami za opóźnienie od 18.07.2023 r.
3.  Wnioski dowodowe:
        - faktura VAT nr 90/2023.
        - umowa cesji.
        - dokumentacja szkody nr PL187923840131.
        - Biegły z zakresu techniki samochodowej i wyceny pojazdów mechanicznych.

===  PRZYKŁADY DLA ODPOWIEDZI NA POZEW === 

1. Pozwany wnosi o oddalenie powództwa w całości.
2. Wnioski dowodowe:
    - Akta szkody nr 123213123/1.
    - Ogólne Warunki Ubezpieczenia Casco Pojazdów (AC).

===============================================
1.  Pozwany wnosi o oddalenie powództwa w całości.
2.  Wnioski dowodowe:
        - Świadkowie: Jan Kowalski, Piotr Kowalski.
        - Podsumowanie zgłosznia szkody.
        - Kosztorys PZU S.A. (ustalenie wysokości szkody).
        - Decyzja pozwanego PZU S.A. z dnia 01.02.2023 r. (ustalenie wysokości odszkodowania).
        - Kosztorys naprawy sporządzy przez warsztat.
        - Biegły z zakresu techniki samochodowej i wyceny pojazdów mechanicznych, na fakt ustalenia kosztów naprawy.

        

UWAGA: Pamiętaj o rozróżnieniu typu dokumentu:
- Jeśli dokument to POZEW → użyj formatu z wartością przedmiotu sporu i roszczeniami powoda
- Jeśli dokument to ODPOWIEDŹ NA POZEW → użyj formatu ze stanowiskiem pozwanego 
"""


# Default paths - can be overridden
DEFAULT_PATHS = {
    "train_documents": "data/training/train/documents",
    "train_summaries": "data/training/train/summaries",
    "test_documents": "data/training/test/documents",
    "test_summaries": "data/training/test/summaries",
    "processed_documents": "data/processed/documents/",
    "processed_summaries": "data/processed/summaries/done",
    "results": "evaluation_results/",
}


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


class UnifiedModelEvaluator(ABC):
    """Unified base class for model evaluation supporting multiple adapter types"""

    def __init__(
        self,
        model_name: str,
        config_name: str,
        adapter_path: Optional[str] = None,
        adapter_type: str = "lora",  # "lora", "prompt_tuning", or "none"
        quantize: bool = True,
        quantize_bits: int = 4,
        results_dir: str = DEFAULT_PATHS["results"],
        enable_thinking: bool = False,
    ):
        self.model_name = model_name
        self.config_name = config_name
        self.adapter_path = adapter_path
        self.adapter_type = adapter_type.lower()
        self.quantize = quantize
        self.quantize_bits = quantize_bits
        self.enable_thinking = enable_thinking

        # Setup results directory
        self.results_dir = Path(results_dir) / self._get_model_family() / config_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Model components
        self.model = None
        self.tokenizer = None

    def _get_model_family(self) -> str:
        """Determine model family from model name"""
        model_lower = self.model_name.lower()
        if "gemma" in model_lower:
            return "gemma"
        elif "qwen" in model_lower:
            return "qwen"
        elif "bielik" in model_lower:
            return "bielik"
        elif "llama" in model_lower:
            return "llama"
        else:
            return "other"

    @abstractmethod
    def _create_chat_messages(self, document_text: str) -> List[Dict[str, str]]:
        """Create chat messages - implemented by model-specific subclasses"""
        pass

    @abstractmethod
    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply chat template - implemented by model-specific subclasses"""
        pass

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration"""
        if not self.quantize:
            return None

        if self.quantize_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.quantize_bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Unsupported quantization bits: {self.quantize_bits}")

    def _load_base_model(self):
        """Load base model and tokenizer"""
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = self._get_quantization_config()

        print(f"Loading base model: {self.model_name}")
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16 if self.quantize else torch.bfloat16,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        # Use appropriate model class
        if "gemma" in self.model_name.lower():
            model_kwargs["attn_implementation"] = "eager"  # More stable for Gemma
            self.model = Gemma3ForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )

    def _load_adapter(self):
        """Load adapter based on adapter type"""
        if not self.adapter_path or not os.path.exists(self.adapter_path):
            print("No adapter to load or adapter path doesn't exist")
            return

        print(f"Loading {self.adapter_type} adapter from {self.adapter_path}")

        if self.adapter_type in ["lora", "prompt_tuning"]:
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

            # Verify adapter loading
            if hasattr(self.model, "get_nb_trainable_parameters"):
                trainable, total = self.model.get_nb_trainable_parameters()
                print(
                    f"Adapter loaded: {trainable:,} trainable parameters ({100*trainable/total:.3f}%)"
                )
        else:
            raise ValueError(f"Unsupported adapter type: {self.adapter_type}")

    def load_model_and_tokenizer(self):
        """Load model, tokenizer, and optional adapter"""
        self._load_base_model()
        self._load_adapter()
        self.model.eval()

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
        word_count = len(text.split())
        estimated_tokens = (
            word_count / 0.75
        )  # Rough approximation: 1 token ≈ 0.75 words
        return estimated_tokens / inference_time

    def generate_summary(self, document_path: str) -> str:
        """Generate summary for a document"""
        document_text = get_doc_text(path=document_path)

        # Create messages using model-specific format
        messages = self._create_chat_messages(document_text)

        # Apply chat template
        text = self._apply_chat_template(messages)

        # Tokenize
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate
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

            # Extract only the generated part
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ][0]

        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()

    def evaluate_dataset(
        self,
        documents_dir: str,
        summaries_dir: str,
        max_docs: Optional[int] = None,
        dataset_name: str = "dataset",
    ) -> tuple:
        """Evaluate on a specific dataset"""
        print(f"Loading model: {self.config_name}")
        with TimingContext() as model_loading_timer:
            self.load_model_and_tokenizer()
        print(f"Model loaded in {model_loading_timer.duration:.2f} seconds")

        results = []
        inference_times = []

        # Get document files
        doc_files = [f for f in os.listdir(documents_dir) if f.endswith(".txt")]

        if max_docs:
            doc_files = doc_files[:max_docs]

        print(f"Evaluating {len(doc_files)} documents from {dataset_name}")

        for doc_file in tqdm(doc_files, desc=f"Evaluating {self.config_name}"):
            doc_path = os.path.join(documents_dir, doc_file)
            reference_path = os.path.join(summaries_dir, doc_file)

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

        return results, inference_times, dataset_name

    def run_evaluation(
        self,
        test_documents_dir: Optional[str] = None,
        test_summaries_dir: Optional[str] = None,
        processed_documents_dir: Optional[str] = None,
        processed_summaries_dir: Optional[str] = None,
        max_docs: Optional[int] = None,
        datasets_to_evaluate: Literal["test", "all"] = "test",
    ) -> Dict:
        """Run evaluation on specified datasets"""

        all_results = {}

        if datasets_to_evaluate == "test":
            test_docs_dir = test_documents_dir or DEFAULT_PATHS["test_documents"]
            test_sums_dir = test_summaries_dir or DEFAULT_PATHS["test_summaries"]

            results, times, name = self.evaluate_dataset(
                test_docs_dir, test_sums_dir, max_docs, "test_set"
            )
            all_results["test"] = (results, times, name)

        elif datasets_to_evaluate == "all":
            proc_docs_dir = (
                processed_documents_dir or DEFAULT_PATHS["processed_documents"]
            )
            proc_sums_dir = (
                processed_summaries_dir or DEFAULT_PATHS["processed_summaries"]
            )

            results, times, name = self.evaluate_dataset(
                proc_docs_dir, proc_sums_dir, max_docs, "all_set"
            )
            all_results["all"] = (results, times, name)

        # Save results for each dataset
        final_metrics = {}

        for dataset_key, (
            results,
            inference_times,
            dataset_name,
        ) in all_results.items():
            if results:
                # Calculate average metrics
                avg_metrics = {
                    "config_name": self.config_name,
                    "model_name": self.model_name,
                    "adapter_path": self.adapter_path,
                    "adapter_type": self.adapter_type,
                    "dataset": dataset_name,
                    "avg_rougeL": sum(r["rouge-l"] for r in results) / len(results),
                    "avg_bleu": sum(r["bleu"] for r in results) / len(results),
                    "avg_meteor": sum(r["meteor"] for r in results) / len(results),
                    "avg_bertscore": sum(r["bertscore"] for r in results)
                    / len(results),
                    "avg_inference_time_seconds": sum(inference_times)
                    / len(inference_times),
                    "min_inference_time_seconds": min(inference_times),
                    "max_inference_time_seconds": max(inference_times),
                    "total_inference_time_seconds": sum(inference_times),
                    "avg_tokens_per_second": sum(
                        r["tokens_per_second"] for r in results
                    )
                    / len(results),
                    "document_count": len(results),
                }

                # Save detailed results
                output_file = (
                    self.results_dir / f"{dataset_key}_evaluation_results.json"
                )
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "config_name": self.config_name,
                            "model_info": {
                                "model_name": self.model_name,
                                "adapter_path": self.adapter_path,
                                "adapter_type": self.adapter_type,
                            },
                            "summary_metrics": avg_metrics,
                            "detailed_results": results,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                print(
                    f"{dataset_name} evaluation complete. Processed {len(results)} documents."
                )
                print(f"Results saved to {output_file}")

                final_metrics[dataset_key] = avg_metrics

        return final_metrics


class StandardModelEvaluator(UnifiedModelEvaluator):
    """Evaluator for standard models (Qwen, Llama, Bielik, etc.)"""

    def _create_chat_messages(self, document_text: str) -> List[Dict[str, str]]:
        """Create chat messages for standard models"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Streść poniższy dokument:\n{document_text}"},
        ]

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply chat template for standard models"""
        try:
            if "qwen3" in self.model_name.lower() and self.enable_thinking:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            else:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception as e:
            print(f"Chat template failed, using fallback format: {e}")
            return f"System: {system_prompt}\n\nUser: Streść poniższy dokument:\n{messages[1]['content']}\n\nAssistant: "


class GemmaModelEvaluator(UnifiedModelEvaluator):
    """Evaluator for Gemma models (no system role support)"""

    def _create_chat_messages(self, document_text: str) -> List[Dict[str, str]]:
        """Create chat messages for Gemma models (combine system and user)"""
        return [
            {
                "role": "user",
                "content": f"{system_prompt}\n\nStreść poniższy dokument:\n{document_text}",
            },
        ]

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply chat template for Gemma models"""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"Chat template failed, using fallback format: {e}")
            return f"User: {messages[0]['content']}\n\nAssistant: "


def create_evaluator(
    model_name: str,
    config_name: str,
    adapter_path: Optional[str] = None,
    adapter_type: str = "lora",
    **kwargs,
) -> UnifiedModelEvaluator:
    """Factory function to create appropriate evaluator based on model name"""
    if "gemma" in model_name.lower():
        return GemmaModelEvaluator(
            model_name=model_name,
            config_name=config_name,
            adapter_path=adapter_path,
            adapter_type=adapter_type,
            **kwargs,
        )
    else:
        return StandardModelEvaluator(
            model_name=model_name,
            config_name=config_name,
            adapter_path=adapter_path,
            adapter_type=adapter_type,
            **kwargs,
        )


def evaluate_trained_models(
    training_output_dir: str,
    model_name: str,
    adapter_type: str = "lora",
    datasets_to_evaluate: Literal["test", "all"] = "test",
    max_docs: Optional[int] = None,
    **kwargs,
):
    """Evaluate all trained models from a training pipeline output directory"""
    training_output_path = Path(training_output_dir)

    if not training_output_path.exists():
        print(f"Training output directory not found: {training_output_dir}")
        return []

    # Find all config directories
    config_dirs = [
        d
        for d in training_output_path.iterdir()
        if d.is_dir() and d.name != "__pycache__"
    ]

    if not config_dirs:
        print(f"No trained model configs found in {training_output_dir}")
        return []

    print(f"Found {len(config_dirs)} trained configurations to evaluate:")
    for config_dir in config_dirs:
        print(f"  - {config_dir.name}")

    all_results = []

    # Evaluate each trained configuration
    for config_dir in config_dirs:
        config_name = config_dir.name

        # Check if this directory contains a trained model
        adapter_config_file = (
            "adapter_config.json" if adapter_type == "lora" else "adapter_config.json"
        )
        if not (config_dir / adapter_config_file).exists():
            print(f"Skipping {config_name} - no {adapter_config_file} found")
            continue

        print(f"\n{'='*60}")
        print(f"EVALUATING: {config_name}")
        print(f"{'='*60}")

        # Create evaluator for this config
        evaluator = create_evaluator(
            model_name=model_name,
            config_name=config_name,
            adapter_path=str(config_dir),
            adapter_type=adapter_type,
            **kwargs,
        )

        # Run evaluation
        metrics_dict = evaluator.run_evaluation(
            datasets_to_evaluate=datasets_to_evaluate, max_docs=max_docs
        )

        # Collect metrics from all datasets
        for dataset_key, metrics in metrics_dict.items():
            all_results.append(metrics)

    # Create comparison report
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Save comparison to CSV
        comparison_path = (
            Path(DEFAULT_PATHS["results"]) / f"{adapter_type}_models_comparison.csv"
        )
        results_df.to_csv(comparison_path, index=False)
        print(f"\nDetailed comparison saved to: {comparison_path}")

        return all_results
    else:
        print("❌ No successful evaluations completed.")
        return []


def evaluate_specific_models(model_configs: List[Dict], **kwargs):
    """Evaluate specific model configurations"""
    all_results = []

    for i, config in enumerate(model_configs):
        print(f"\n{'='*60}")
        print(f"EVALUATING: {config['config_name']} ({i+1}/{len(model_configs)})")
        print(f"{'='*60}")

        evaluator = create_evaluator(**config)
        metrics_dict = evaluator.run_evaluation(**kwargs)

        # Collect metrics from all datasets
        for dataset_key, metrics in metrics_dict.items():
            all_results.append(metrics)

    # Create comparison report
    if all_results:
        results_df = pd.DataFrame(all_results)
        comparison_path = (
            Path(DEFAULT_PATHS["results"]) / "specific_models_comparison.csv"
        )
        results_df.to_csv(comparison_path, index=False)
        print(f"\nComparison saved to: {comparison_path}")
        return all_results

    return []


def main():
    model_configs = [
        {
            "model_name": "speakleash/Bielik-4.5B-v3.0-Instruct",
            "config_name": "bielik_base",
            "adapter_path": None,
            "adapter_type": "none",
            "quantize": True,
        },
        {
            "model_name": "Qwen/Qwen3-4B",
            "config_name": "qwen3_base",
            "adapter_path": None,
            "adapter_type": "none",
            "quantize": True,
        },
        # Add more configurations as needed
    ]

    evaluate_specific_models(
        model_configs=model_configs,
        datasets_to_evaluate=["test"],
    )


if __name__ == "__main__":
    main()
