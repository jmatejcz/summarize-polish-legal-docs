from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from src.utils.read_files import DocumentReader
import pprint

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
)

model_name = "speakleash/Bielik-11B-v2.3-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
)

doc_reader = DocumentReader("data/documents/I_C_148022_odpowiedz_na_pozew.pdf")
I_C_148022_odpowiedz_na_pozew = doc_reader.read_all_pages_string()
pprint.pprint(I_C_148022_odpowiedz_na_pozew)
# print(len(I_C_148022_odpowiedz_na_pozew))


def build_prompt(text: str) -> str:
    return f"""### Instruction:
Streść poniższy tekst.

### Input:
{text}




### Response:
pozwany wnosi o:
- oddalenie powództwa w całości 
- dopuszczenie dowodu z dokumentu - akt szkody nr <numer>
- dopuszczenie dowodu z zeznań świadka - <imie> 
- dopuszczenie i przeprowadznie dowodu - cennik wynajmu aut publikowany na stronach internetowych wypozyczalni aut z 2022r.
- dopuszczenie dowodu z opinii biegłego z  zakresu techniki samochodowej i ruchu drogowego​
- podnosi zarzut braku legitymacji czynnej powoda z uwagi na nieważność umowy cesji
- pozwany kwestionuje wysokość odszkodowania z tytułu naprawy pojazdu (stawka roboczogodziny zastosowaną przez zakład naprawczy i obniżył ją do 95zł netto.)
- pozwany zweryfikował stawkę dobową najmu ze 108zł netto do 65zł netto, a także okres najmu z 63 do 18 dni


### Input:
{text}

### Response:"""


def summarize(text: str, max_new_tokens=400) -> str:
    prompt = build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(inputs["input_ids"].shape)
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

# print("Max context length:", model.get_po,,,sition_embeddings)
print(summarize(text))




