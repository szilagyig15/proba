Machine Learning

1. Modellek:
   1. automatikusan tanuljon az adatokból 
   2. nem mondod meg explicit, hogy mit csináljon a program és magától fogja megtanuli az inputok és outputokból
   3. nem az volt a lényeg, hogy szofisztikált lgyen, ha elég sok adatod van
   4. csoportok:
      1. Supervised -  van megfigyeléshalmaz és outputok
      2. Unsupervised - szegmentálja az sokaságot (homogá csoportokat csinál)
      3. Reinforcemnet learning
2. Features (x -  inputok, független változó) és Target (függő változó) változók:
3. Cél:
   1. Olyan algorimusokat hoztak létre,amik maguktól tanulnak
   2. Akkor jó, ha jól teljesít új adatokra
4. Feladat:
   1. y=a+b*x és ráillesztünk lineáris regressziót 
5. Lineéris regresszió
   1. folytonos változót jelzünk előre
   2. y=a+b*x
   3. a - tengelymetszet
   4. b - meredekség
6. Polinomiális Regresszió
   1. overfittinget eredményez
   2. jobb illeszkedést eredményez
7. Modell illeszkedése
   1. loss funcion: predictív meg igaz változók között mi az eltérés ezt méri
   2. Mean Squared Error (MSE)
   3. Mean Absolute Error (MAE)
8. Prediction, Generalization & Overfitting
   1. predicton: becslést az a target változóra, új adatállományban
   2. generalization: jól teljesít új adatokon
   3. egyensúly!
9. Bias and variance:
   1. bias: mennyire tér el az előrejelzés a tény adattól (mennyire tér el)
   2. variance: mennyire változna az előrejelzés, ha másak a training adatok
   3. na nő a bias a variance csökken