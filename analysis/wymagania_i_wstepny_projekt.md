# Wnioski

## Wybór zadania
Na podstawie analizy literatury i analizy natury dokumentów doszedłem do wniosku, zadanie generowania w zamyśle takim jaki miałem na początku 
jest zadaniem chyba niemożliwym do zrealizowania w pojedynkę w ramach pracy magistersiej. Wynika to z tego, że nie ma na ten moment dostępnych
rozwiązań, które byłyby w stanie generować treści natury sądowej/ prawniczej. Przygotowania takiego rozwiązania wymagałoby wytrenowania modelu
zdolnego do interpretacji przepisów prawnych, zrozumienia okoliczności i wydedukowania na jakie przepisy należy się powołać itp.
W związku z tym skłaniam się w stronę streszczania.

## Przewidywane trudności 
Trudności ze streszczeniem dokumentów prawnych wymienione w literaturze wskazują, że jest to trudne zadanie, ale nie wszystkie one tyczą się
dokumnetów sądowych. Często podnoszony argument o tym, że na podstawie streszczenia abstrakcyjnego nie da się w pełni poprawnie zinterpretować
dokumentu, jest prawdziwy, ale uważam, że głównie w dokumentach typu: ustawy, rozporządzenia, umowy cywilnoprawne, natomiast pisma sądowe 
nie są aż tak zależe od doboru słów  i raczej nie trzeba się zastanawiać nad ich interpretacją. Kolejnym problem zdaje się być długość dokumentów.
O ile pozwy czy odpowiedzi na pozew, gdzie dużo jest załączników do dowodów potrafią mieć nawet 50 stron, o tyle większość z tego zazwyczaj to własnie
dowody, których tak naprawdę nie trzeba streszczać. Wynika to z natury dowodów, jeżeli prawnik będzie chciał się z nim zaznajomić to i tak musi go w 
całości obejrzeć / przeczytać. Odrzucając dowody zostają wnioski/zarzuty i uzasadnienie, które łącznie zazwyczaj nie przekraczają 10 stron.

## Wybór metod i narzędzi 
Biorąc pod uwagę strukturę dokumentów i formę streszczeń zdecydowanie zależeć będzie mi bardziej na streszczeniu abstrakcyjnym. Narzędziami do generowania takich streszczeń sa wyłącznie modele AI, tutaj pod uwagę biorę jedynie modele offline z uwagi na poufną naturę dokumentów. Preferowałbym również modele mniejsze i średnie (DistilBERT, Pointer-Generator, Legal-BERT-Small, BERT-Base, PEGASUS, BART), jako że docelowo rozwiązanie miałoby działac na sprzęcie konsumenckim, natomiast jeżeli nie będą one dawać sobie rady przetestuje również modele dużę (BIELIK.AI, Longformer, BART-Large).
Jako metryki do oceny streszczeń wykorzystam popularne w tym zakresie metryki z rodziny ROUGE, a subiektywną metryką będzie ocena profesjonalistów. Żeby oszczędzić ich czasu, wstępną ocenę zgodności z faktami przeprowadzę sam, a im zostawię do oceny jedynie składnię i sposób podania informacji.  


## Wstępny podział na podzadania
Biorąc pod uwagę analizę literatury, naturę dokumentów i dostępne rozwiązania stawiam przed swoją pracą następujące wymagania:
- stworzenie małej bazy dokumentów, zawierającej możliwie dużo streszczeń przygotowanych przez profesjonalistów
- zanonimizowanie tych dokumentóœ
- Wstępne przestestoawnie poszczególnych modeli i zdecydowanie się na te najbardziej obiecujące.
- Dotrenowanie wybranych modeli 
- Przetestowanie i porównanie streszczeń generowanych przez oryginalne modele z tymi dotrenowanymi 
- Porównanie z modelami online oraz innymi rozwiązaniami z innych prac na publicznych orzeczeniach sądów 

