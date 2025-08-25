# LITERATURA, state of the art

## Streszczanie
streszczanie dokumentów można podzielić na:
- ekstrakcyjne(extractive) - wyciąganie fragmentów tekstu z dokumentu, które uznamy za ważne. Extractive Summarization of Text Using Supervised and Unsupervised Techniques [https://ieeexplore.ieee.org/document/9537883]
- abstrakcyjne(abstractive) - w oparciu o zrozumienie dokumentu, piszemy nowe zdania, które dobrze oddają esencję dokumentu.

Przed erą AI, streszczanie skupiało się na 1 metodzie, natomiast w przypadku dokumentów prawnych, którą są długie, 
złożone i napisane formalnym językiem, takie podejście nie sprawdzało się najlepiej, ponieważ streszczenia takie ciągle musiały być długie
i formalne co jest troche zaprzeczeniem streszczenia. Jako zaletę tego podejścia, niekórzy argumentują, że dokumenty prawne są skomplikowane,
więc nie da się bez ekstrakcyjnego streszczenia w pełni oddać ich znaczenia. Tutaj warto zaznaczyć różnice w dokumentach prawnych. Ja w swojej pracy
chcę skupić się na dokumentach prawniczych - to są dokumenty pisane przez strony sporu sądowego oraz sam sąd w trakcie sprawy. Są one tylko 
podzbiorem dokumentów prawnych, tak więc widząc, że jakieś rozwiązanie działa na dokumentach prawnych (legal documents), nie oznacza
to, że rozwiązania te działają na dokumentach sądowych/prawniczych. Często tak jak wymieniam poniżej, rozwiąznaia działają w obszarze np. dokumentów 
legislacyjnych, które znacznie różnią się strukturą i treścią. W związku z tym proszę zwrócić uwagę na dobór słów w dalszej części tekstu.
prawne != prawnicze/sądowe

### Metody 
Teraz krótko opiszę metody do generowania streszczeń, które pojawiają się w pracach: 
- Metody ekstrakcyjne, nienadzorowane,  do stosowania ogólnego (streszczania kazdego rodzaju dokumentów), np.: 
    - Reduction - Metoda ta polega na identyfikacji i usuwaniu mniej istotnych fragmentów tekstu, koncentrując się na zachowaniu kluczowych informacji.
    - LexRank - algorytm oparty na grafach, który ocenia znaczenie zdań na podstawie ich centralności w grafie podobieństw.
    - LSA - Analiza ukrytych wymiarów semantycznych polega na dekompozycji macierzy częstości słów w dokumentach za pomocą technik takich jak SVD 
    (Singular Value Decomposition). Pozwala to na identyfikację ukrytych struktur semantycznych i wybór zdań najlepiej reprezentujących główne 
    tematy tekstu.
    - PacSum - Metoda ta wykorzystuje model BERT do reprezentacji zdań, a następnie buduje graf podobieństw między nimi.
- Metody ekstrakcyjne, nadzorowane,  do stosowania ogólnego, np.:
    - SummaRunner - Jest to model rekurencyjnej sieci neuronowej (RNN), który traktuje streszczanie jako zadanie sekwencyjnej klasyfikacji. 
    Model uczy się przewidywać, które zdania powinny zostać uwzględnione w streszczeniu.
    - BERTSum - Rozszerzenie modelu BERT dostosowane do zadań streszczania ekstrakcyjnego. Model ten klasyfikuje każde zdanie jako 
    odpowiednie lub nieodpowiednie do streszczenia.
- Metody ekstrakcyjne, nadzorowane, specjalne dla dokumentów prawnych:
    - LetSum - system, który wykorzystuje analizę strukturalną dokumentu, identyfikując kluczowe sekcje i ekstraktując z nich istotne informacje. 
    Metoda ta opiera się na analizie częstości występowania terminów oraz ich rozmieszczeniu w dokumencie, co pozwala na wyodrębnienie najważniejszych 
    zdań do streszczenia.
    - KMM (K-mixture Model) - Polega na budowie modelu rozkładu terminów w dokumencie, bazując na modelu K-mieszanin(probabilistyczny model 
    statystyczny stosowany do reprezentowania złożonych rozkładów danych poprzez kombinację kilku prostszych rozkładów, najczęściej rozkładów normalnych).
    Model ten jest następnie wykorzystywany do generowania streszczenia poprzez wybór zdań, które najlepiej reprezentują główne tematy dokumentu
    - Gist - Proces rozpoczyna się od reprezentacji każdego zdania za pomocą różnych cech, takich jak długość zdania, pozycja w dokumencie czy częstość 
    występowania terminów. Następnie wykorzystuje trzy modele: wielowarstwowy perceptron (MLP), gradientowe drzewa decyzyjne oraz LSTM, które klasyfikują
        zdania pod kątem ich przydatności do streszczenia.
- Metody abstrakcyjne, opierają się o modele AI:
    - Pointer-Generator - Model ten łączy mechanizm generowania z mechanizmem wskazywania (pointer), co pozwala na kopiowanie słów bezpośrednio z tekstu 
    źródłowego oraz generowanie nowych słów. Dzięki temu model radzi sobie z problemem słów spoza słownika (OOV) i redukuje powtarzanie fraz.
    [https://arxiv.org/abs/1704.04368]
    - BERTSumABS - Rozszerzenie modelu BERT dostosowane do zadań streszczania abstrakcyjnego. Model ten wykorzystuje pretrenowany BERT jako enkoder 
    w architekturze typu encoder-decoder, co pozwala na generowanie streszczeń poprzez dekodowanie zakodowanych reprezentacji tekstu źródłowego.
    - Pegasus - Model pretrenowany specjalnie do zadań streszczania, wykorzystujący technikę Gap Sentence Generation (GSG). 
    W trakcie pretrenowania model uczy się przewidywać brakujące zdania w tekście, co zbliża zadanie pretrenowania do rzeczywistego 
    zadania streszczania.[https://huggingface.co/docs/transformers/model_doc/pegasus]
    - BART -  Model typu encoder-decoder, który łączy cechy modeli autoregresyjnych i autoenkoderów. BART jest pretrenowany 
    poprzez rekonstrukcję zniekształconych wejść, co czyni go efektywnym w zadaniach generacyjnych, takich jak streszczanie.
    - Longformer - Model zaprojektowany do przetwarzania długich dokumentów poprzez wprowadzenie mechanizmu uwagi (attention), który skaluje się liniowo 
    z długością sekwencji. Longformer łączy lokalną uwagę okienkową z globalną uwagą, co pozwala na efektywne przetwarzanie dokumentów 
    zawierających tysiące tokenów. [https://huggingface.co/docs/transformers/model_doc/longformer]

### Metryki
Metryki używane do oceny streszczeń, które pojawiają się najczęściej:

- ROUGE-N - Mierzy nakładanie się n-gramów (ciągów n słów) między podsumowaniem a tekstem referencyjnym. Na przykład, ROUGE-1 analizuje 
pojedyncze słowa, a ROUGE-2 pary słów.
- ROUGE-L - Opiera się na najdłuższym wspólnym podciągu między podsumowaniem a tekstem referencyjnym, uwzględniając strukturę zdań.
- ROUGE-S - Analizuje pary słów w ich oryginalnej kolejności, nawet jeśli nie są one sąsiadujące w tekście.
Rodzina metryk ROUGE koncentruje się na powierzchniowym dopasowaniu słów, co może nie w pełni oddawać semantycznej zgodności między tekstami.

- BERTScore - wykorzystuje zaawansowane modele językowe, takie jak BERT, do oceny podobieństwa między tekstami na poziomie semantycznym. 
Zamiast porównywać dokładne dopasowanie słów, BERTScore oblicza podobieństwo między osadzonymi wektorami słów, 
co pozwala na uchwycenie kontekstu i znaczenia. BERTScore wykazuje wyższą korelację z ocenami ludzkimi w porównaniu 
z tradycyjnymi metrykami, takimi jak ROUGE, zwłaszcza w zadaniach wymagających głębszego zrozumienia kontekstu.

### Prace

Nie znalazłem, żadnej pracy operującej na polskich dokumentach prawnych w zakresie streszczania lub generowania. 
Znalazłem natomiast pracę o klasyfikacji orzeczeń sądów [https://www.mdpi.com/1424-8220/22/6/2137] napisaną między innymi przez profesora 
PW - Roberta Nowaka. Bazuje ona na zbiorze zanonimizowanych orzeczeń publikowanych przez sądy. Niestety zawiera ona jedynie wyroki bez 
wcześniejszych pism z obu stron. Idąc tym tropem szukałem publicznych, rządowych baz, pytając rówież osoby z branży, jednak nie istnieje 
żadna publiczna baza zawierająca polskie pisma prawnicze.

W pracy Automatic Summarization of Legal Text [https://studenttheses.uu.nl/handle/20.500.12932/34261] 
wytrenowano wlasny model NLP do dokumentow w języku holenderskim. Jako metryk użyto streszczeń porównawczych i metyk ROUGE. Przeprowadzono też 
eksperyment, w którym ludzie studiujący prawo mieli ocenić jakość tych streszczań. Najpierw studenci czytali i zaznajamiali się z dokumentem, 
potem oceniali streszczenia, w tym te wygrnerowane przez model i oceniali w skali od 1 do 10 trafność i czytelność. Wyniki pokazywały, że
model często nie uwzględnia w streszczeniu ważnych informacji, oraz popełnia blędy składniowe.

W pracy Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation [https://aclanthology.org/2022.aacl-main.77.pdf]
naukowcy wskazują podobne problemy, które ja również napotykam - trudność w użyciu nadzorowanego uczenia, dlatego iż ciężko o duże ilości streszczeń 
napisanych przez prawników, co sprawia, że zbiory danych są bardzo małe. Dodatkowo wskazują oni na to, że streszczania dokumentów prawniczych są dłuższe niż 
w przypadku innych dokumentów, ponieważ są one bardziej skomplikowane. W pracy użyto zbiorów wyroków sądów najwyższych w Indiach i Wielkiej Brytanii
oraz ich streszczeń napisanych przez profesjonalistów. Praca porówuje metody ekstrakcyjne i abstrakcyjne, zarówno nadzrorowane jak i nie. Naukowcy 
wskazują kolejny problem, który pojawia się przy długich dokumentach, mianowicie nie mieszczą się one w wejściu modelu, w związku z tym proponują 
oni 2 podejścia: segmentacja tekstu oraz najpierw streszczenie ekstrakcyjne i potem streszczenie abstrakcyjne na podstawie ekstrakcyjnego.
Wyniki Przeprowadzonych testów pokazują podobne wartości metryk ROUGE i BertScore między streszczaniami ekstrakcyjnymi(nadzorowanymi i nie) 
i abstrakcyjnymi, natomiast eksperci preferują streszczenia ekstrakcyjnie, jako że są dokładniejsze.

W pracy z 10 października 2024, czyli bardzo świeżej- Advancing Legal Document Summarization: Introducing an Approach Using a Recursive Summarization 
Algorithm [https://link.springer.com/article/10.1007/s42979-024-03277-3], naukowcy wprowadzają nowatorskie podejście do streszczenia długich dokumentów
prawnych - RecSumm. [https://github.com/Saloni-sharma29/Recursive-summarization]. System opiera się na podziale tekstu na częsci, podsumowaniem ich
oddzielnie, a następnie iteracyjnego poprawiania kolejnych streszczeń na podstawie poprzednich. na koniec łączy wszystko w całość. Według naukowców 
system ten przewyższył w metrykach wszystkie z popularnych modeli LLM, w tym te przystosowane do dziedziny - Chat GPT, Gemini, Pegasus.

## Generowanie

Niewiele jest prac w temacie generowania dokumentów sądowych, w pracach z tym związanych autorzy zwracają uwagę, że do napisania poprawnego
dokumentu sądowego potrzebna jest wiedza na temat przepisów, ich interpretacji, zrozumienia okoliczności , dobre uargumentowanie itp.
co jest zadaniem na ten moment za trudnym dla algorytmów. AI  w sądownictwie jest częściej wykorzystywane do przydzielania spraw, wydawanie
standardowych zarządzeń, analizie materiału dowodowego, czy orzekania w sprawach prostych i powtarzalnych. Nie znalazlem jednak żadnego narzędzia 
lub pracy o koknretnie wygenerowaniu pełnego dokumentu sądowego. Warto zaznaczyć, że nawet jeżeli jakiś model rozumie prawo i potrafi je 
interpretować w jednym kraju, to nie znaczy, że będzie to potrafił w innym, np. Polsce, jako, że prawo i jego interpretacje różnią się między 
krajami. Najwięszke doświadczenie z wprowadzaniem AI do sądownictwa mają Chiny, gdzie zaczęto ją wykorzystywać już w 2017r. W Unii, która jest nam 
bliższa jest to mocno utrudnione przez AI Act.

Praca również nowa bo z 9 października 2024 Unlocking authentic judicial reasoning: A Template-Based Legal Information Generation 
framework for judicial views[https://www.sciencedirect.com/science/article/pii/S0950705124008669]. 

Knowledge-Based Legal Document Assembly [https://arxiv.org/abs/2009.06611] na podstawie wywiadu z prawnikiem program generuje szablon 
dopasowany do sprawy


	
	
	

