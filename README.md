
## Wstępny projekt

kolejność czytania:

literatura i analiza dziedziny -> dokumenty -> dostępne modele LLM -> wymagania i wstępny projekt


## setup 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```


## setup hugginface

```bash
pip install huggingface_hub
huggingface-hub login
```

Następnie wkleić PAT z huggingface 

### Ewaluacja

Ewaluację modeli można przeprowadzić na pomocą skryptu `evaluate_models.py`

```bash
uv run evaluate_models.py 
```

### Trening
Trening został zaimplementowany w dwóch skryptach:
- `training/prompt_tuning/train_models.py` oraz skrypt bashowy `train_prompt_tuning.sh`, który umożliwia trening metodą prompt tuning wielu modeli za jednym zamachem.

- `training/qlora/train_models.py` oraz `train_qlora.sh` to samo tylko dla metody qlora

s