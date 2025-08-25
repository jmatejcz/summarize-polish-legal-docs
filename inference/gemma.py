from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,
)
import torch
from data_preprocess import system_prompt, get_doc_text

model_name = "google/gemma-3-4b-it"


model = Gemma3ForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Streść poniższy dokument:\n" + get_doc_text(
    path="data/processed/documents/XI_C_80924_odpowiedz_na_pozew.txt"
)

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_prompt,
            }
        ],
    },
    {"role": "user", "content": [{"type": "text", "text": prompt}]},
]


inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

generation = model.generate(**inputs, max_new_tokens=512, do_sample=False)
generation = generation[0][input_len:]

decoded = tokenizer.decode(generation, skip_special_tokens=True)
print(decoded)
