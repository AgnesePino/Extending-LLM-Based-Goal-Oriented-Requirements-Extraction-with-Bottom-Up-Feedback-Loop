# Tuning delle soglie

Due file Python: `__main__.py` prepara i confronti e gestisce i comandi;
`workbook.py` gestisce Excel, annotazioni e calcolo delle soglie.
Ogni preparazione salva un solo Excel in `outputs/<run>/`.

Esegui i comandi dalla cartella principale del progetto, in PowerShell.
Servono le dipendenze del progetto installate nella `.venv`.

## 1. Preparare l'Excel

Per riusare le ricostruzioni bottom-up salvate:

```powershell
.venv\Scripts\python.exe -m experiment_threhold_tuning prepare
```

Servono le baseline in `output/top_down` e le ricostruzioni iniziali del
notebook 02. Il programma riusa soltanto quelle compatibili e dichiara
i dataset esclusi nel foglio **Copertura dataset**. Il calcolo delle cosine
richiede il modello di embedding configurato nella pipeline.

Per generare nuove ricostruzioni uniformi, avvia prima LM Studio con Qwen
configurato per il progetto e usa:

```powershell
.venv\Scripts\python.exe -m experiment_threhold_tuning prepare --source fresh
```

Questa modalita ricostruisce gli HLG dai LLG senza eseguire il ciclo di correzione.
Il percorso dell'Excel viene stampato al termine.

## 2. Annotare e aggiornare

Apri l'Excel. Nei fogli **Confronti** compila le celle gialle:
`equivalent=true/false` e, se utili, le note. Poi salva e chiudi Excel.

```powershell
.venv\Scripts\python.exe -m experiment_threhold_tuning tune
```

Il comando trova il workbook se e l'unico presente. Se ci sono piu workbook,
specifica il file oppure la sua cartella:

```powershell
.venv\Scripts\python.exe -m experiment_threhold_tuning tune "experiment_threhold_tuning/outputs/NOME_RUN"
```

Controlla **Riepilogo tuning** e **Griglia soglie**. La proposta usa solo
i dataset di calibrazione e richiede esempi sia equivalenti sia non equivalenti;
i dataset di validazione restano separati. Il confronto con la ground truth
e diagnostico.

## 3. Confermare per gli esperimenti

Nel foglio **Scelta soglie** inserisci la soglia unica in **B2** e `true`
in **C2**. La riga duplicati riusa la stessa scelta. Salva, chiudi Excel
ed esegui nuovamente `tune`. Il notebook 03 legge la scelta tramite
`confirmed_thresholds`: imposta `THRESHOLD_WORKBOOK` sul file preparato.

Per le opzioni aggiuntive: `python -m experiment_threhold_tuning prepare --help`.
Il comando `consolidate "cartella"` resta disponibile per convertire i vecchi
archivi JSON/CSV senza ricalcolare i confronti.
