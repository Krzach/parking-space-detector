# parking-space-detector

Wykrywanie miejsc parkingowych na podstawie dlugosci postoju samochodu na danym miejscu
Wykorzystac nauczona siec neuronowa yolo do detekcji samochodow
W miare mozliwosci widok od gory
Walidacja tez na obrazach w zlej pogodzie/zimie

pip install --upgrade ultralytics

1. Potencjalne miejsce parkingowe jest wykrywane, jeżeli wykryto samochód w danym miejscu przez określona ilość czasu (liczbę zdjęć/klatek z filmu). Wtedy takie potencjalne miejsce jest zapisywane.
2. Miejsce jest zapisane za pomocą ramki (wielokątu) zdefiniowanej według numerów pikseli na zdjęciu, a obliczonego na podstawie zbioru ramek, o podobnych w granicy X, centroidach.
3. Następnie przy rozpoznawaniu zajętości miejsca uznajemy, że miejsce jest zajęte, jeśli ponad 50% obszaru miejsca jest zajęte przez obiekt rozpoznany jako samochód (czyli ramka lub wielokąt otaczający taki samochód zajmuje Y np. 50% obszaru ramki miejsca).

Transmisje na zywo z parkingami (trzeba kliknąć feed URL by otrzymać link do właściwego streama, który będzie obsłużony w kodzie)
```
https://opencctv.org/cam/17211
https://opencctv.org/cam/17212
https://opencctv.org/cam/17216
```

Aby odpalić obecne rozwiązanie:
1. dostrój parametry na offline_cars_detector.py - dla porządanego zbioru
2. uruchom online_cars_detector.py - z czasem zbierze się odpowiednia liczba obrazów
3. uruchom spots_detector.py i dostrój parametry tak by miejsca zgadzały się możliwie z rzeczywistością
4. TODO: spots_occupacy_detector.py

