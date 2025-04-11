# tratec-tesi
Archivio dei file relativi alla mia tesi in "Bias di genere e traduzione automatica" per il corso di LM in Specialized translation - Translation & technology @ DIT, Università di Bologna, Forlì.
La tesi completa è disponibile [qui](https://amslaurea.unibo.it/id/eprint/32051). 

## Dati
I dati di addestramento e valutazione per i classificatori (EN e IT) si trovano nella cartella `gc_training_data`.

Nella cartella `data` si trovano i dati utilizzati negli esperimenti per addestrare (train.csv, val.csv e TM.csv) e valutare (test.csv) i modelli adattati.

## Codice
Nella cartella `scripts` si trovano gli script Python utilizzati per addestrare i classificatori (uno per lingua) e il rewriter.

## Modelli
La cartella `models` contiene i classificatori in formato Keras e la configurazione del rewriter.

## Valutazione manuale
`manual_eval` contiene i file .tsv con la valutazione manuale dei risultati di tutti gli esperimenti di riscrittura. 
