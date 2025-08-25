
## Dokumenty

Myślę, że warto o krótki opis natury dokumentów, które będą danymi w tej pracy, jako że ktoś spoza 
branży prawa może nie wiedzieć jak one wyglądają. Tak jak pisałem w literaturze dokumenty sądowe / prawnicze 
to nie to samo co dokumenty prawne. W skład dokumentów prawnych wchodzą np. ustawy, rozporządzenia, umowy cywilnoprawne,
to są dokumenty których interpretacja zależy ściśle od użytych słów, kontekstu i odwołań, przez co streszczanie takich
dokumentów jest szczególnie trudne - zły dobór słów może zmienić interpretację. 

Inną kategoria, która mnie interesuje to dokumenty sądowe i prawnicze. Można je podzielić śledząc przebieg sprawy. 
Sprawy zaczynają się od pozwów pisanych przez prawników, potem druga strona wystosowuje odpowiedź na pozew, 
w którym odnosi się do zarzutów, następnie obie strony mogą wymieniać się dowolną ilością pism przygotowawczych, w 
których odpowiadają drugiej stronie na zarzuty / argumenty / dowody. Te dokumenty są często dosyć długie przez załączanie
do nich dowodów w różniej postaci, np. skanów umów, aktów, zeznań itp. O ile pozwy są dosyć powtarzalne i kancelarie moją własne
wzory do różnych typów spraw, np. o ubezpieczenie, to już odpowiedzi są zależne mocno od okoliczności i tego co przedstawi druga strona.

Sąd w czasie sprawy wydaje wezwania, zawiadomienia, postanowienia, zarządzenia, które są zazwyczaj prostymi dokumentami organizacyjnymi.
Na koniec sąd wydaje orzeczenie: wyrok, nakaz zapłaty, postanowienie. 

Od orzeczenia sądu można się odwołać do wyższej instancji apelacją. Wtedy sprawa zaczyna się "od nowa" w sądzie wyższej instancji.

### Bazy Danych

Nie istnieje baza danych wszystkich pism które są pisane na drodze sądowej, istnieje natomiast baza wyroków. Na dokumentach z tej bazy
mogę porównać wyniki z innymi pracami. [https://orzeczenia.ms.gov.pl/].
Mam zamiar stworzyć własną małą bazę danych pism sądowych - pozwów, odpowiedz, pism przygotowawczych, apelacji, wezwań, zarządzeń. 
Dokumenty będą pogrupowane według spraw. Do tego możliwie dużo streszczeń referencyjnych do tych dokumentów napisanych przez profesjonalistów.

### Anonimizacja

Interesują mnie narzędzia do anonimiacji dostępne za darmo oraz offline, wśród takich znalazłem:
- spaCy + MedSpaCy - biblioteka NLP z opcją wykrywania danych osobowych
- Presidio - narzędzie od Microsoft 
- Faker - Generowanie fikcyjnych danych zamiast rzeczywistych
- Scrubadub - proste narzędziw w Pythonie


### Streszczenia

Prawnicy poproszeni o pisanie streszczeń to dokumentów, tworzą raczej krótkie streszczenia zawierające tylko kluczowe informacje, takie informacje najbardziej ich interesują:
 - o co składający pismo wnosi /czy uznaje powództwo, czy wnosi o odrzucenie pozwu?​
- jakie składa wnioski dowodowe /świadkowie , dokumenty , biegły sądowy?​
- fakty sporne/co kwestionuje/ i fakty bezsporne/czego nie kwestionuje? 

Przykładowo:

pozwany wnosi o oddalenie powództwa w całości.​

składa wnioski:​

dopuszczenie dowodu z dokumentu - akt szkody​

dopuszczenie dowodu z zeznań świadka jana kowalskiego​

dopuszczenie dowodu z opinii biegłego z  zakresu techniki samochodowej i ruchu drogowego​

Fakty sporne to wysokość stawki roboczogodziny, pozwany obniżył stawkę rbh z 290 zł na 120 zł. ​

### Uwagi 

Pobieram dokumenty z jednej kancelarii, i o ile istnieje pewien formalizm pisania takich pism, o tyle różnią się one troche między kancelariami.
Dodatkowo każda kancelaria specjalizuje sie w pewnych sprawach(np. karnych, lub o rozwody), nie oznacza to ,że brak ejst tam innych spraw,
ale spraw z innch dziedzin będzie zdecydowanie mniej.