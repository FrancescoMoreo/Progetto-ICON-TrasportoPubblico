Repository per il progetto di Ingegneria della conoscenza 2024-2025 realizzato da:
- Francesco Moreo (matricola: 760453)

## Installazione

Per installare il progetto è necessario clonare la repository con il seguente comando:

```bash
git clone https://github.com/FrancescoMoreo/Progetto-ICON-TrasportoPubblico.git 
```

Ed installare SWI-Prolog in base al sistema operativo in uso dal sito ufficiale: [SWI-Prolog](https://www.swi-prolog.org/download/stable)

## Struttura del progetto

- ```implementazione```: contiene il codice sorgente del progetto
- ```documentazione```: contiene la documentazione del progetto
- ```progettazione```: contiene il dataset scelto per l'apprendimento, i dataset di pre-processing e altri file generati a runtime

## Utilizzo

1. Installare le dipendenze necessarie:

```bash
pip install -r progettazione/requirements.txt
```

  - Controllare le loro versioni lanciando ```progettazione/check_version_lib.py```

    ```bash
    python progettazione/check_version_lib.py
    ```

2. Lanciare il file ```main.py```:

```bash
python implementazione/main.py
```
