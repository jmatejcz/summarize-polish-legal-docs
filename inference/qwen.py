from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch
from peft import PeftModel
from data_preprocess import system_prompt, get_doc_text

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=quantization_config, device_map="auto"
)
prompt_tuned = PeftModel.from_pretrained(
    model,
    "./qwen2.5_prompt_tuning/",  # <-- folder where you saved adapter
    device_map="auto",  # keep it on the right device
)


prompt = "Streść poniższy dokument:\n" + get_doc_text(
    path="data/processed/documents/XI_C_80924_pozew.txt"
)
messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = prompt_tuned.generate(**model_inputs, max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
