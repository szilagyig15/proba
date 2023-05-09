# Introduction to Machine Learning Concepts

Modellezni a valóságot és prediktálni, ami alapján döntéseket lehet hozni.
A machine learning direkt emberi beavatkozás nélkül tanul: nem mondjuk meg explicite az algoritmusnak, hogy mit
csináljon, hanem beadunk egy nagy outputot, és a mintázatokat megtanulja felismerni.

Modellek:
1. supervised
2. unsupervised (children :)): nincs megadva semmilyen eredmény, ott csak egy ponthalmaz van

Speciális fogalmak:
1. feature: inputadatok
2. target: függő változó

A machine learning céljai:
1. egy olyan algoritmust hozzon létre, amelyek maguktól tanulnak, és jól prediktáljanak

Akkor jó egy machine learning modell, ha jól teljesít az új adatokon.

A lineáris regresszió a legegyszerűbb machine learning modell: itt folytonos változókat regresszálunk.

Polinomiális regresszió:
- overfittinget eredményez

Model fitting
- training set-re illesztünk modellt
- loss function-t minimalizáljuk az illesztésnél
- generalization: jól teljesít-e még nem látott adatokon
- túlillesztés: a patern helyett a zajt figyeli meg

Bias and variance trade-off:
- bias: mennyire tér el az előrejelzés a training adaton, azaz mennyire tér el a training adatokon
  - ha nagy a bias, akkor nagy az eltérés
- variance: ha más adatokat adok be, akkor mennyire fog fluktuálni a modell vagy mi a f*sz

Hányadfokú polinomot illesztünk, ez egy hiperparaméter.
Model fitting: keressük azokat a paramétereket, amik minimalizálják a loss function-t

Splitting: szétosztjuk az adatokat training set-re és test set-re
- ezzel megakadályozzuk az overfittinget
- a training set-re fitteljük a modellt
- a test set-en teszteljük, hogy mennyire jól illeszkedik
- lehet egy harmadik rész is, a validation set

Cross validation (k-fold cross validation):
- a training set-en traineljük a modellt, a test set-en pedig "ellenőrizzük"
- a cross validation félreteszi az adatok egy részét tesztelésre (test set)
- a maradék részt szétdobja k-szor training set-re és validation setre
- ezekből jön k db error function
- a k db error function átlagát adja vissza a modell error function-jeként