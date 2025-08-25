from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from src.utils.read_files import DocumentReader

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)

model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=quantization_config,
)

doc_reader = DocumentReader("data/documents/I_C_148022_odpowiedz_na_pozew.pdf")
page_generator = doc_reader.read_next_page()
text = next(page_generator)
text += next(page_generator)
text += next(page_generator)
text += next(page_generator)
text += next(page_generator)
text += next(page_generator)
text += next(page_generator)
text += next(page_generator)
# text += next(page_generator)

# print(text)


def build_prompt(text: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Streść poniższy tekst:

{text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def summarize(text: str, max_new_tokens=400) -> str:
    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # print(inputs["input_ids"].shape)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded.split("### Response:")[-1].strip()
    return response

# print("Max context length:", model.get_position_embeddings)
print(summarize(text))



