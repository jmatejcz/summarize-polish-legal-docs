
## Wstępny projekt

kolejność czytania:

literatura i analiza dziedziny -> dokumenty -> dostępne modele LLM -> wymagania i wstępny projekt


## setup 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
export PYTHONPATH=/path/to/repo/polish-legal-docs-summarization-and-generation
```


## setup hugginface

```bash
pip install huggingface_hub
huggingface-hub login
```
then paste the PAT from huggingface