from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from data_preprocess import system_prompt, get_doc_text

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)


model_name = "togethercomputer/Llama-2-7B-32K-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
)
prompt = "Streść poniższy dokument:\n" + get_doc_text(
    path="data/processed/documents/XI_C_80924_pozew.txt"
)
input_context = system_prompt + prompt
input_ids = tokenizer.encode(input_context, return_tensors="pt").to(model.device)
input_len = input_ids.shape[-1]
output = model.generate(input_ids, max_new_tokens=512, temperature=0.7)
output_text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
print(output_text)
