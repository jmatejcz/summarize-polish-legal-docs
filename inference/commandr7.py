import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from data_preprocess import system_prompt, get_doc_text

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)


model_id = "CohereLabs/c4ai-command-r7b-12-2024"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
)

prompt = "Streść poniższy dokument:\n" + get_doc_text(
    path="data/processed/documents/XI_C_80924_pozew.txt"
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt},
]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, return_tensors="pt"
)

model_inputs = tokenizer([input_ids], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
