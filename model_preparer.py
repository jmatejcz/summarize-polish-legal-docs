import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Gemma3ForCausalLM,
)
import gc


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


UWAGA: Pamiętaj o rozróżnieniu typu dokumentu:
- Jeśli dokument to POZEW → użyj formatu z wartością przedmiotu sporu i roszczeniami powoda
- Jeśli dokument to ODPOWIEDŹ NA POZEW → użyj formatu ze stanowiskiem pozwanego 
"""

SYSTEM_PROMPT_WITH_EXAMPLES = (
    SYSTEM_PROMPT
    + """
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
"""
)


class BaseModelPrepare(ABC):

    def __init__(
        self,
        model_name: str,
        quantize: bool = True,
        quantize_bits: int = 4,
    ):
        self.model_name = model_name
        self.quantize = quantize
        self.quantize_bits = quantize_bits
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if not self.tokenizer:
            raise ValueError("Tokenizer not laoded")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = self._get_quantization_config()

        print(f"Loading base model: {self.model_name}")
        self.model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }

        if quantization_config:
            self.model_kwargs["quantization_config"] = quantization_config

    @abstractmethod
    def _create_chat_messages(self, document_text: str) -> List[Dict[str, str]]:
        """Create chat messages - implemented by model-specific subclasses"""
        pass

    @abstractmethod
    def create_training_chat_messages(
        self, document_text: str, target_text: str
    ) -> List[Dict[str, Any]]:
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

    @abstractmethod
    def _load_base_model(self):
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


class CommonModelPrepare(BaseModelPrepare):
    """For models with standard chat structure"""

    def __init__(self, model_name: str, quantize: bool = True, quantize_bits: int = 4):
        super().__init__(model_name, quantize, quantize_bits)
        self._load_base_model()

    def _create_chat_messages(self, document_text: str) -> List[Dict[str, str]]:
        """Create chat messages for standard models"""
        return [
            {"role": "system", "content": SYSTEM_PROMPT_WITH_EXAMPLES},
            {"role": "user", "content": f"Streść poniższy dokument:\n{document_text}"},
        ]

    def create_training_chat_messages(
        self, document_text: str, target_text: str
    ) -> List[Dict[str, str]]:
        base_chat = self._create_chat_messages(document_text=document_text)
        base_chat.append(
            {
                "role": "assistant",
                "content": target_text,
            }
        )
        return base_chat

    def _load_base_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **self.model_kwargs
        )

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:

        if "qwen3" in self.model_name.lower():
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )


class GemmaModelPrepare(BaseModelPrepare):
    """For gemma models"""

    def __init__(self, model_name: str, quantize: bool = True, quantize_bits: int = 4):
        super().__init__(model_name, quantize, quantize_bits)
        self._load_base_model()

    def _create_chat_messages(self, document_text: str):
        """Create chat messages for Gemma models (combine system and user)"""
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_WITH_EXAMPLES,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Streść poniższy dokument:\n{document_text}",
                    }
                ],
            },
        ]

    def create_training_chat_messages(
        self, document_text: str, target_text: str
    ) -> List[Dict[str, Any]]:
        base_chat = self._create_chat_messages(document_text=document_text)
        base_chat.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            }
        )
        return base_chat

    def _load_base_model(self):
        self.model = Gemma3ForCausalLM.from_pretrained(
            self.model_name,
            **self.model_kwargs,
        )

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply chat template for Gemma models"""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def create_preparer(
    model_name: str,
    **kwargs,
) -> BaseModelPrepare:
    """Factory function to create appropriate evaluator based on model name"""
    if "gemma" in model_name.lower():
        return GemmaModelPrepare(
            model_name=model_name,
            **kwargs,
        )
    else:
        return CommonModelPrepare(
            model_name=model_name,
            **kwargs,
        )
