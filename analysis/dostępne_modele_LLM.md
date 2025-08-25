## Dostępne LLMy

Nie wymienię tutaj oczywiście wszystkich, jedynie te które mogą być dobre w interesującej mnie dziedzinie.

### Dostępne jedynie online:
1.  Chat-GPT 4 - ogromny model, dokładny rozmiar nie został ujawniony, wersja 3 miała 175 miliardów parametrów. Bardzo popularny, o bardzo szerokim zatosowaniu
2.  Cloude - podobnie co Chat GPT, bardziej nastawiony na poprawność polityczną

### Dostępne Offline:
1. BERT (https://github.com/google-research/bert) w wersjach:
    - BERT Base (110 milionów parametrów) (https://huggingface.co/google-bert/bert-base-uncased)
    - BERT Large (340 milionów parametrów) (https://huggingface.co/google-bert/bert-large-uncased)
    - BERT-Multilingual (179 milionów parametrów) (https://huggingface.co/google-bert/bert-base-multilingual-cased)
    - DistilBERT (67 milionów parametrów) (https://huggingface.co/distilbert/distilbert-base-uncased)
    - BERTSumABS (109 milionów parametrów) - Wykorzystuje standardowy BERT jako enkoder oraz 6-warstwowy dekoder Transformer,
        inicjalizowany losowo (https://huggingface.co/vituong/bertsumabs1 ??) [https://blog.paperspace.com/extractive-text-summarization-with-bertsum/]

2. BIELIK.AI - polski model językowy, opracowany na bazie podelu Mistral-7B, dostępny w wersjach:
    - 7 miliardów parametrów (https://huggingface.co/speakleash/Bielik-7B-v0.1)
    - 11 miliardów parametrów ((https://huggingface.co/speakleash/Bielik-11B-v2))

3. Legal-BERT - model BERT dostosowany do analizy tekstów prawnych
    - BASE (110 milionów parametrów) (https://huggingface.co/nlpaueb/legal-bert-base-uncased)
    - SMALL (36 milionów parametrów) https://huggingface.co/nlpaueb/legal-bert-small-uncased
    - są też edycje dostosowane do prawa w dziedzinach:
        - CONTRACTS-BERT-BASE - amerykańskie kontrakty 
        - EURLEX-BERT-BASE - legislacja Unii Europejskiej
        - ECHR-BERT-BASE - orzeczenia Europejskiego Trybunału Praw Człowieka

3. PEGASUS - posiada wiele wersji trenowanych do specyficznych zadań, tą najbardziej odpowiednią do zadania streszczania 
    wydaje się model: 
    -   PEGASUS-XSum (568 milionów parametrów) (https://huggingface.co/google/pegasus-xsum)

4. BART (Bidirectional and Auto-Regressive Transformers)  model opracowany przez Facebook AI, dostępny w wersjach:
    - BART-Base (139 milionów parametrów) – (https://huggingface.co/facebook/bart-base)
    - BART-Large (406 milionów parametrów) – (https://huggingface.co/facebook/bart-large)
    - BART-Large-CNN (406 milionów parametrów) – dostrojony do streszczania artykułów informacyjnych z 
        wykorzystaniem zbioru danych CNN/DailyMail – (https://huggingface.co/facebook/bart-large-cnn)
    - BART-Large-XSum (406 milionów parametrów) – dostrojony do streszczania z wykorzystaniem zbioru danych 
        XSum – (https://huggingface.co/facebook/bart-large-xsum)

- Longformer to model oparty na architekturze transformera, zaprojektowany do efektywnego przetwarzania
długich sekwencji tekstu. Tradycyjne modele transformerowe mają ograniczenia w przetwarzaniu długich
    dokumentów ze względu na kwadratową złożoność operacji self-attention w stosunku do długości sekwencji.
    Longformer wprowadza mechanizm uwagi, który skaluje się liniowo z długością sekwencji, umożliwiając
    przetwarzanie dokumentów zawierających tysiące tokenów. [https://arxiv.org/abs/2004.05150]
    - Longformer-Base-4096 (149 milionów parametrów) (https://huggingface.co/allenai/longformer-base-4096)
    - Longformer-Encoder-Decoder (162 milionów parametrów) (https://huggingface.co/allenai/led-base-16384)

- Pointer-Generator Network to architektura sieci neuronowej zaprojektowana w celu poprawy jakości streszczania tekstów,
    łącząca podejścia ekstrakcyjne i abstrakcyjne. Tradycyjne modele abstrakcyjne generują nowe zdania, co może prowadzić
    do błędów faktograficznych, natomiast modele ekstrakcyjne kopiują fragmenty tekstu źródłowego, co może skutkować 
    brakiem płynności. Pointer-Generator Network umożliwia zarówno generowanie nowych słów, jak i kopiowanie bezpośrednio
    z tekstu źródłowego, co pozwala na tworzenie bardziej dokładnych i spójnych streszczeń. [https://arxiv.org/abs/1704.04368]
    [https://github.com/jiminsun/pointer-generator], liczba parametrów zalezy od implementacji, która nie jest ustalona z góry,
    mniej wiecej od 8 do 15 milionów parametrów
