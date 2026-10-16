# Catena multi-agente LLM per l'allineamento delle API orientato agli obiettivi

## Panoramica

Questo repository estrae gerarchie di obiettivi dalla documentazione software
utilizzando la Goal-Oriented Requirements Engineering (GORE) e i Large Language
Model. Conserva la pipeline top-down originale proposta in *Evaluating LLM-Based
Goal Extraction in Requirements Engineering: Prompting Strategies and Their
Limitations* e aggiunge una fase separata di validazione bottom-up per il
confronto sperimentale sviluppato in questo progetto.

## Architettura: catena multi-agente LLM

![Architettura del sistema](Architecture.png)

La pipeline originale utilizza una catena LLM top-down fissa. Ogni fase riceve
l'output strutturato di quella precedente:

1. **Pre-elaborazione della documentazione (opzionale):** converte un README non
   strutturato o un documento dei requisiti in una descrizione del progetto in
   linguaggio naturale.
2. **Identificazione degli attori:** estrae i ruoli degli utenti finali che
   interagiscono con il sistema.
3. **Estrazione degli High-Level Goal:** identifica gli obiettivi funzionali
   strategici degli attori estratti.
4. **Estrazione dei Low-Level Goal:** scompone l'insieme completo degli HLG in
   azioni funzionali concrete. Tutti gli HLG vengono forniti insieme in una
   singola chiamata di generazione, come nell'implementazione originale.
5. **Mapping delle API:** associa gli LLG risultanti agli endpoint API quando è
   disponibile la relativa documentazione.

La generazione di attori, HLG e LLG può utilizzare il meccanismo
generator-critic originale. Quando è abilitato, il critic assegna un punteggio e
una critica all'output generato. Se il punteggio è inferiore alla soglia di
qualità configurata, la critica viene restituita al generatore per un nuovo
tentativo, fino al numero massimo di iterazioni.

Il ciclo bottom-up implementato in questo repository è un'estensione e non fa
parte della baseline top-down originale. Parte da un risultato top-down
persistito, ricostruisce gli HLG a partire dai relativi LLG e può confermare,
correggere, rimuovere o rigenerare branch incoerenti. L'output top-down originale
rimane separato, così da poter valutare indipendentemente le due configurazioni.

Il percorso originale di generazione e reflection degli HLG top-down non applica
una deduplicazione deterministica basata sul nome normalizzato dell'HLG. Rimane
invariato: le decisioni sulla ridondanza semantica introdotte da questa
estensione appartengono al critic HLG bottom-up.

## Flusso effettivo dei notebook

L'esperimento è diviso in tre notebook con responsabilità separate. Il notebook
di valutazione non richiama i modelli e non modifica gli output della pipeline.

| Notebook | Input | Operazioni effettive | Output |
| --- | --- | --- | --- |
| `01_pipeline_execution_top_down_only.ipynb` | `ALL_GROUNDTRUTHS` e documentazione associata | Genera descrizione quando è presente `link-readme`, quindi attori, HLG e LLG; ogni stadio usa generazione Zero-shot e reflection Few-shot | Un JSON per ogni dataset completato in `output/top_down/` |
| `01_pipeline_execution_bottom_up_only.ipynb` | JSON strutturati in `output/top_down/` | Carica la gerarchia senza rieseguire l'estrazione iniziale, valida e corregge i branch con discovery disabilitata | Risultato finale, oppure artefatto di errore, in `output/top_down_bottom_up/`; trace in `output/bottom_up_iterations/` |
| `02_experimental_evaluation_top_down_vs_bottom_up.ipynb` | Coppie complete top-down/top-down+bottom-up e ground truth | Estrae i goal salvati, calcola metriche semantiche finali e per iterazione e produce grafici; non genera goal e non esegue critic | CSV, JSON e PNG in `output/evaluation/top_down_vs_bottom_up/` |

Il percorso sperimentale corrente termina alla valutazione di HLG e LLG. Il
mapping degli LLG sugli endpoint API fa parte dell'architettura generale del
repository, ma non viene eseguito da questi tre notebook e non entra nelle
metriche riportate di seguito.

## Ciclo bottom-up corrente di validazione e correzione

La baseline usa la logica originale di `src.extraction.extractor`, `src.utils`
e `src.self_critique.refine_response`: prompt dell'allegato, parsing originale,
soglia 8,5 e 3 tentativi. La baseline usa Qwen/Ministral; il bottom-up usa
Qwen per costruzione e riparazioni e l'ensemble Ministral/Bonsai/Gemma per le
decisioni HLG/LLG. Zero-shot, critic attivo e
`ENABLE_HLG_DISCOVERY = False` costituiscono la configurazione sperimentale
corrente: il run riportato in questo README non esegue il critic di copertura
globale e non cerca nuovi HLG mancanti.
La reflection originale e quella focalizzata per le riparazioni bottom-up sono
nello stesso `src.self_critique.refine_response`, ma usano simboli distinti: la
baseline conserva i nomi e i prompt originali, mentre l'estensione usa il
prefisso `BottomUp`/`bottom_up_`, con soglia 8,0 e 5 tentativi. Analogamente,
`src.extraction.extractor` conserva i generatori originali e aggiunge
`generate_bottom_up_low_level_goals` per il solo contesto di riparazione. Il
generatore HLG e i client restano condivisi.

Il comportamento operativo, la matrice dei critic, i dettagli del voting e i
limiti riscontrati nel codice sono descritti nel
[README bottom-up](src/bottom_up/README_bottom_up.md). Una iterazione analizza
tutti i branch attivi prima di applicare le correzioni; nell'iterazione iniziale
sono attivi tutti i branch. La convergenza è un accordo interno dei critic,
non una garanzia di corrispondenza alla ground truth.

La configurazione registrata negli artefatti dell'esperimento è:

| Parametro | Valore del run |
| --- | --- |
| Generatore | `qwen/qwen3.8-27b` tramite Groq |
| Valutatori HLG/LLG | Ensemble locale Ministral + Bonsai + Gemma tramite LM Studio |
| Valutatore discovery | Ensemble locale Ministral + Bonsai + Gemma tramite LM Studio |
| Discovery HLG | disabilitata |
| Voting dei critic HLG/LLG | Few-shot: 9 esempi HLG e 7 esempi LLG; 3 modelli con lo stesso input, load–evaluate–unload sequenziale; quorum 2/3 per HLG e veto LLG motivato per la rigenerazione |
| Prompt discovery | Few-shot con 3 esempi e condizione di discovery esplicita, votato separatamente dai tre modelli |
| Generazione delle riparazioni | Zero-shot + reflection Few-shot con 4 esempi, soglia 8/10, massimo 5 tentativi |
| Memoria dei critic | per progetto, 4 scambi, 2 riepiloghi di stato, budget storico di 1.000 caratteri |
| Limite del ciclo | 25 iterazioni nel notebook |
| Pacing dataset bottom-up | gruppi da 3; 60 s tra dataset e 120 s dopo ogni gruppo |
| Correzioni per iterazione | massimo 2 rewrite HLG, oppure 2 rimozioni HLG, oppure 2 rigenerazioni LLG |
| Client del notebook | timeout 120 secondi, 1 retry per Qwen/Groq e Ministral/LM Studio |

### Obiettivi e perimetro dell'input

La fase bottom-up ha un unico obiettivo principale: **validare e correggere** la
gerarchia prodotta dalla baseline, rimuovendo ridondanze effettive e riparando le
decomposizioni HLG-LLG incoerenti.

Il notebook usa i nomi presenti in `ALL_GROUNDTRUTHS` soltanto come allowlist per
selezionare i file top-down. I goal annotati della ground truth non vengono
passati al reconstructor, ai critic o ai generatori bottom-up; vengono letti
soltanto in seguito dal notebook di valutazione.

La discovery degli HLG mancanti è un modulo separato e opzionale, disattivato per
impostazione predefinita. Può essere abilitato nel notebook impostando
`ENABLE_HLG_DISCOVERY = True`; soltanto in questa modalità il ciclo cerca nella
descrizione del progetto intenzioni a livello di attore non rappresentate
nell'insieme corrente di HLG.

Non ripete l'estrazione iniziale di attori, HLG o LLG. Riceve in input la
descrizione del progetto persistita dalla fase top-down e gli oggetti strutturati
HLG e LLG prodotti dalla stessa fase. Il perimetro della discovery coincide
quindi con le informazioni contenute in tale descrizione: sezioni del README,
evidenze OpenAPI, issue, interviste o altri artefatti non inclusi non possono
attualmente produrre nuovi requisiti.

Ogni LLG contiene già il parent strutturato `high_level_associated` emesso dalla
pipeline top-down originale. La fase bottom-up raggruppa gli LLG tramite questo
oggetto incorporato, senza confrontare i nomi, applicare fallback sugli attori o
riscrivere la relazione. Prima dell'avvio del ciclo, la baseline viene rifiutata
se un parent incorporato non è un membro esatto di
`structuredHighLevelGoals`. La baseline top-down non viene mai sovrascritta; il
risultato combinato e il trace completo delle iterazioni vengono salvati
separatamente in `output/top_down_bottom_up/` e
`output/bottom_up_iterations/`.

### Algoritmo di esecuzione

Durante ogni iterazione, l'orchestratore esegue le seguenti operazioni:

1. costruisce la vista dei branch direttamente dal parent top-down incorporato
   in ciascun LLG;
2. ricostruisce un HLG diagnostico a partire dagli LLG di ogni branch attivo,
   senza mostrare al reconstructor l'HLG parent originale;
3. confronta semanticamente l'HLG originale con la ricostruzione, poi con la
   descrizione completa e con i sibling. Il risultato espone
   `reconstruction_relationship` (`EQUIVALENT`, `PARTIAL_OVERLAP`,
   `MATCHES_SIBLING`, `DIFFERENT` o `NOT_RECONSTRUCTABLE`) e il flag
   `requires_llg_evaluation`;
4. invoca il critic LLG per ogni HLG mantenuto, anche quando la ricostruzione è
   `EQUIVALENT`: l'equivalenza aggregata non dimostra infatti che ogni singolo
   LLG sia supportato, necessario e non ridondante. `REWRITE` e `REMOVE` non
   valutano la vecchia decomposizione. Il critic LLG riceve
   anche il verdetto HLG aggregato: una relazione materialmente non equivalente
   non può essere annullata da un `KEEP` LLG e richiede la rigenerazione mirata
   della decomposizione;
5. soltanto quando la discovery è abilitata e tutti i branch correnti sono stati
   confermati, esegue un prompt separato di copertura sull'intero progetto,
   considerando la descrizione, l'insieme corrente di HLG e le intenzioni
   esplicite o implicate dal workflow per gli attori esistenti;
6. applica un insieme limitato di correzioni alla gerarchia corrente e avvia una
   nuova iterazione;
7. termina quando tutti i branch sono confermati, quando non rimangono azioni
   eseguibili oppure quando viene raggiunto il limite di iterazioni. Se la
   discovery è abilitata, richiede inoltre che non risultino HLG mancanti.

I due critic di validazione, HLG e LLG, condividono una conversazione limitata
per ciascun progetto. Quando la discovery è abilitata, anche il critic opzionale
di copertura usa la stessa conversazione. Dopo ogni iterazione, l'orchestratore registra le
aggiunte, rimozioni o sostituzioni di HLG e le aggiunte o rimozioni di LLG
effettivamente applicate a ciascun parent.

Dalla seconda iterazione in poi, il riepilogo della gerarchia risultante
dall'iterazione precedente viene incluso esplicitamente in ogni richiesta al
critic. La prima iterazione riceve soltanto un riepilogo della gerarchia
iniziale. Generatori, ricostruzione, valutazione top-down e progetti differenti
non condividono questa memoria. Il contesto conservato è compatto: contiene al
massimo quattro scambi recenti del valutatore e conserva due riepiloghi degli
aggiornamenti, ma invia solo l'ultimo. Il budget di 1.000 caratteri limita gli
scambi, non il riepilogo o il prompt corrente, anziché riprodurre
integralmente i prompt precedenti.

Ciascun prompt del critic include esempi few-shot specifici per il proprio
compito. Il prompt HLG calibra `KEEP`, `REWRITE` e `REMOVE`; quello LLG confronta
decomposizioni valide e associabili alle API con difetti sostanziali di
copertura. Il prompt globale opzionale distingue le intenzioni degli attori
realmente mancanti dalle varianti più ristrette di obiettivi già coperti.

Ogni decisione HLG/LLG del valutatore bottom-up utilizza tre chiamate sequenziali
indipendenti, una per `mistralai/ministral-3-14b-reasoning`, una per
`prism-ml/bonsai-27b` e una per `google/gemma-4-12b-qat`. Le decisioni HLG
mantengono la maggioranza 2 su 3. Per gli LLG, se tutti e tre i voti sono validi,
anche un solo `REGENERATE_LOW_LEVEL_GOALS` corredato dalle prove strutturate
richieste attiva la riparazione: due `KEEP` generici non possono cancellare un
difetto concreto identificato dal terzo critic. Se manca un voto valido, torna
ad applicarsi il quorum ordinario e un pareggio resta inconclusivo.
I tre modelli ricevono esattamente lo stesso prompt e lo stesso snapshot della
memoria. Ogni modello viene caricato in LM Studio, produce il proprio voto e
viene scaricato prima di caricare il successivo. Per il primo critic il quorum riguarda
anche le rimozioni HLG, senza requisito di unanimità, mantenendo i controlli
sulla base della rimozione e sul sibling coprente. Il quorum viene calcolato
sulla decisione (`KEEP`, `REWRITE`, `REMOVE` e analoghe), non sulla terna
relazione/decisione/routing. Se la maggioranza del critic HLG sceglie `KEEP` ma
diverge sulla relazione, viene conservato un voto rappresentativo della
maggioranza, privilegiando quello che richiede il controllo LLG. Ogni voter assegna inoltre
all'artefatto corrente un punteggio di qualità da 0 a 5 usando esempi specifici:
solo 4 e 5 sono compatibili con una decisione di passaggio, mentre 0–3 richiedono
una correzione. Il critic LLG è più restrittivo: `KEEP_LOW_LEVEL_GOALS` richiede
5/5. Un voto il cui punteggio contraddice la decisione viene scartato prima
dell'aggregazione. Il risultato conserva un voto rappresentativo, non la media
degli score né l'unione delle evidenze. Soltanto il verdetto aggregato viene scritto
nella memoria e nel trace dell'iterazione. Per impostazione predefinita, una
risposta individuale non valida dispone di tre tentativi complessivi (due retry);
`BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS` accetta valori da 1 a 3.

Il routing HLG–LLG segue questo albero decisionale:

| Relazione diagnosticata | Stato del parent | Decisione | Critic LLG |
|---|---|---|---|
| `EQUIVALENT` | documentato e valido | `KEEP` | sì, audit indipendente di ogni LLG |
| `PARTIAL_OVERLAP` | valido | `KEEP` | sì |
| `PARTIAL_OVERLAP` | formulato male | `REWRITE` | no |
| `MATCHES_SIBLING` | interamente ridondante | `REMOVE` | no |
| `MATCHES_SIBLING` | distinto e valido | `KEEP` | sì |
| `DIFFERENT` | valido | `KEEP` | sì |
| qualsiasi, anche `EQUIVALENT` | materialmente errato o incompleto, ma recuperabile | `REWRITE` | no |
| qualsiasi | intenzione completamente non documentata | `REMOVE` | no |
| `NOT_RECONSTRUCTABLE` | valido ma senza LLG | `KEEP` | sì, per generare gli LLG |

Le riscritture degli HLG e le rigenerazioni degli LLG utilizzano Qwen in
Zero-shot seguito dalla reflection Few-shot specifica del bottom-up: quattro
esempi, soglia `8/10` e fino a cinque tentativi. Questo critic generativo è
distinto dai critic HLG/LLG con voting e usa una sola valutazione per tentativo.
La reflection può restituire l'ultimo risultato anche se resta sotto soglia dopo
l'esaurimento dei tentativi; il branch viene comunque rivalutato dal ciclo
successivo. L'estrazione degli attori non viene ripetuta e la configurazione di
generazione non viene ereditata dalla baseline. Le generazioni dovute alla
discovery seguono lo stesso percorso soltanto quando il modulo è abilitato.

Il valore predefinito dell'orchestratore è di 5 iterazioni, mentre il notebook
bottom-up corrente imposta esplicitamente un limite sperimentale di 25. Un branch
confermato viene ignorato finché la struttura degli HLG rimane invariata.
L'aggiunta, la rimozione o la riscrittura di un HLG annulla tali conferme e
attiva un nuovo audit completo dei branch nell'iterazione successiva. I branch
usano ID tecnici derivati dall'intero HLG strutturato e da un indice di
occorrenza. Questi ID servono esclusivamente a coordinare le decisioni: non
implicano che nomi uguali siano equivalenti o che nomi diversi rappresentino
obiettivi distinti.

| Decisione del valutatore | Aggiornamento della gerarchia corrente | Crea un nuovo branch? |
|---|---|---|
| `CONFIRM_BRANCH` | Mantiene il branch finché la struttura degli HLG non cambia | No |
| `REWRITE_ORIGINAL_HIGH_LEVEL_GOAL` | Chiama il percorso generativo bottom-up, sostituisce un HLG e ne rigenera gli LLG | Sostituzione, non discovery netta |
| `REMOVE_ORIGINAL_HIGH_LEVEL_GOAL` | Elimina il branch; se ridondante, richiede consolidamento degli LLG nel target coprente | No; può rigenerare un branch mantenuto |
| `REGENERATE_LOW_LEVEL_GOALS` | Chiama il generatore LLG bottom-up per lo stesso HLG parent | No |
| `MISSING_HIGH_LEVEL_GOALS_FOUND` | In modalità discovery, chiama il percorso generativo bottom-up per HLG e LLG | Sì |
| `EVALUATION_INCONCLUSIVE` | Lascia invariata la gerarchia corrente e registra un warning | No |

Le correzioni alla gerarchia degli obiettivi vengono ordinate deliberatamente:
in una singola iterazione vengono applicate al massimo due riscritture HLG; solo
se non viene applicato alcun rewrite possono essere eseguite fino a due
rimozioni HLG; solo in assenza di rewrite e rimozioni possono essere rigenerati
fino a due branch LLG. Le correzioni indipendenti della categoria selezionata
sono applicate insieme dopo aver raccolto i verdetti del pass. La
discovery opzionale degli HLG mancanti non ha un limite numerico quando è
abilitata: ogni gap distinto e documentato restituito dal critic di copertura
viene generato una volta che la gerarchia è stabile. Un branch esistente può
essere rigenerato o riscritto al massimo due volte. Questi limiti sono
protezioni ingegneristiche contro cicli di correzione ripetuti, non soglie
ottimizzate empiricamente.

Le decomposizioni LLG, sia quelle generate per un nuovo HLG sia quelle rigenerate
per un branch esistente, non hanno un limite numerico fisso: al generatore viene
richiesta una decomposizione completa e non ridondante delle interazioni
funzionali distinte e documentate, successivamente controllata per individuare
obiettivi vuoti, duplicati o associati in modo errato. Le azioni di riscrittura e
rimozione hanno la precedenza sulle aggiunte di HLG mancanti, perché modificano
lo spazio degli obiettivi rispetto al quale viene valutata la copertura globale.

### Discovery opzionale

**Non nella configurazione predefinita.** Con `ENABLE_HLG_DISCOVERY = False` il
ciclo esegue esclusivamente validazione e correzione e non chiama il critic di
copertura globale. Impostando il flag a `True`, la copertura viene valutata solo
dopo la conferma di tutti i branch correnti e può restituire tutti i gap distinti
e documentati a livello di attore. Un gap può
essere esplicito oppure dedotto da almeno due passaggi coerenti del workflow o da
responsabilità dell'attore. Quando viene accettato, i normali generatori HLG e
LLG del percorso bottom-up creano un nuovo branch, che viene poi sottoposto allo
stesso audit dei branch esistenti. Soltanto con questa opzione il processo può
aggiungere un HLG realmente nuovo invece di limitarsi a confermare, riscrivere,
rimuovere o rigenerare parti della baseline.

Dal codice emergono alcuni vincoli della discovery corrente, ancora da misurare
sperimentalmente:

- la discovery si basa su un'unica descrizione consolidata del progetto, anziché
  su una raccolta indicizzata di passaggi sorgente;
- la discovery è limitata agli attori già rappresentati nell'insieme corrente di
  HLG;
- non conserva una mappa evidence-to-goal e il giudizio di copertura resta
  interamente basato sul modello;
- non è presente un controllo deterministico che dimostri che un HLG aggiunto
  copra davvero un requisito prima assente senza duplicare un branch esistente;
- il controllo globale di discovery usa una singola chiamata Ministral ogni
  volta che la gerarchia torna stabile, oltre alle eventuali
  chiamate di generazione e al nuovo audit dei branch.

### Risultati sperimentali con discovery disabilitata

L'ultimo esperimento salvato usa esplicitamente
`ENABLE_HLG_DISCOVERY = False`: valuta soltanto la capacità del bottom-up di
**validare e correggere i branch HLG-LLG già presenti nella baseline**. Il critic
di copertura globale non viene invocato, la decisione
`MISSING_HIGH_LEVEL_GOALS_FOUND` non è raggiungibile e nessun nuovo branch può
essere aggiunto tramite discovery. Di conseguenza, questi risultati non misurano
la capacità di recuperare requisiti o HLG assenti.

Il run comprende 11 dataset e usa tre voter indipendenti per ogni decisione dei
critic HLG e LLG, con quorum 2 su 3. Tutti gli 11 cicli convergono con
`ALL_BRANCHES_CONFIRMED`; in questa configurazione significa che tutti i branch
esistenti sono stati confermati o corretti, non che la gerarchia sia completa
rispetto a eventuali requisiti mancanti. I cicli completano da 3 a 8 iterazioni,
quindi nessuno raggiunge il limite di 25. Otto dataset conservano almeno un
warning nel risultato (18 warning complessivi): convergenza non significa
assenza di anomalie registrate durante le correzioni. La dodicesima baseline,
*Calculator*, non entra nel confronto: il notebook bottom-up ha salvato un
artefatto di errore perché un parent incorporato negli LLG non era presente
nell'insieme degli HLG strutturati. Il notebook di valutazione esclude gli output con
`bottomUpError` o privi delle collezioni finali.

#### Metodo di valutazione

Il notebook `02_experimental_evaluation_top_down_vs_bottom_up.ipynb` associa i
file tramite il campo `name` e include un dataset soltanto se trova baseline,
risultato bottom-up completo e ground truth. Per la baseline legge
`structuredHighLevelGoals`/`structuredLowLevelGoals`; per il feedback legge
`finalHighLevelGoals`/`finalLowLevelGoals`. Da ogni oggetto usa il campo `name`,
con fallback a `description`; non valuta attori, descrizioni complete, mapping
API o razionali dei critic.

La metrica usa `bert-base-uncased` con mean pooling e similarità coseno. Trova un
matching uno-a-uno tra goal generati e di riferimento mediante algoritmo
ungherese, conserva soltanto le coppie con similarità almeno `0.65` e calcola:

- soft Precision = somma delle similarità accettate / goal generati;
- soft Recall = somma delle similarità accettate / goal di riferimento;
- F1 = media armonica delle due quantità.

Le risorse NLTK vengono inizializzate dal notebook, ma la configurazione
effettiva usa `preprocess=False`: i testi entrano in BERT senza stemming,
lemmatizzazione o rimozione delle stopword. Le sintesi seguenti sono
macro-medie: ogni dataset ha lo stesso peso, indipendentemente dal numero di
goal. Le macro-medie finali sono:

| Livello | Configurazione | Precision | Recall | F1 | Goal totali |
| --- | --- | ---: | ---: | ---: | ---: |
| HLG | Top-down | 0.4112 | 0.7216 | 0.5019 | 104 |
| HLG | Top-down + bottom-up | 0.4359 | 0.7155 | 0.5210 | 96 |
| LLG | Top-down | 0.3264 | 0.5113 | 0.3840 | 439 |
| LLG | Top-down + bottom-up | 0.3221 | 0.5030 | 0.3779 | 436 |

Per gli HLG il bottom-up aumenta la precisione di `0.0247`, riduce la recall di
soli `0.0061` e porta la F1 da `0.5019` a `0.5210` (`+0.0191`, circa `+3.8%`).
La F1 migliora in 5 dataset, rimane invariata in 4 e peggiora in 2; il numero
totale di HLG diminuisce di 8 (`-7.7%`). Per gli LLG, invece, precision, recall e
F1 diminuiscono rispettivamente di `0.0043`, `0.0083` e `0.0061` (F1 circa
`-1.6%`). La F1 migliora comunque in 6 dataset, resta invariata in 1 e peggiora
in 4, ma il forte peggioramento di *Assegno Unico Universale* (`-0.0990`) rende
negativo il risultato macro; il numero complessivo di LLG cambia di appena 3
unità (`-0.7%`).

Rispetto ai valori precedentemente documentati con un solo voter, il nuovo
assetto produce correzioni molto meno aggressive:

| Livello | Un solo voter: delta F1 | Voto 2 su 3: delta F1 | Un solo voter: delta goal | Voto 2 su 3: delta goal |
| --- | ---: | ---: | ---: | ---: |
| HLG | +0.0795 (0.5504 → 0.6299) | +0.0191 (0.5019 → 0.5210) | -35 (105 → 70) | -8 (104 → 96) |
| LLG | +0.0532 (0.3768 → 0.4300) | -0.0061 (0.3840 → 0.3779) | -129 (457 → 328) | -3 (439 → 436) |

La maggioranza riduce quindi le rimozioni e protegge meglio la recall HLG: la
perdita passa da `-0.0437` con un voter a `-0.0061` con tre voter. Questa
prudenza, però, riduce anche il guadagno di precisione e non migliora la qualità
LLG aggregata nel nuovo run. Non si può attribuire causalmente la differenza al
solo numero di voter: il vecchio snapshot comprendeva 12 dataset, quello nuovo
ne comprende 11 (manca *Calculator*), e sono cambiati anche i file top-down, la
ground truth e la logica della pipeline. Per misurare isolatamente l'effetto del
voting occorre rieseguire entrambe le configurazioni sugli stessi input, con lo
stesso codice e gli stessi seed. Il confronto non è inoltre un confronto tra
discovery attiva e disattiva: il nuovo snapshot documenta esclusivamente il
percorso con discovery disabilitata.

Come previsto dalla configurazione, nessuno dei trace salvati contiene una
decisione `MISSING_HIGH_LEVEL_GOALS_FOUND`. Questa assenza non è un risultato del
critic: deriva dal fatto che la discovery non è stata eseguita. Le misurazioni
dettagliate della sola validazione/correzione comprendono metriche assolute,
delta per dataset, sintesi macro, andamento per iterazione e due grafici. Sono
salvate in `output/evaluation/top_down_vs_bottom_up/` nei file
`top_down_vs_bottom_up_metrics.csv`, `top_down_vs_bottom_up_deltas.csv`,
`top_down_vs_bottom_up_macro_summary.csv`, `iteration_metrics.csv`,
`iteration_macro_summary.csv`, `iteration_metric_trend.png`,
`iteration_metric_trend_by_dataset.png` ed `evaluation_summary.json`.

Nei grafici iterativi, l'iterazione 0 è la baseline top-down. La media al passo
N include soltanto i dataset che hanno effettivamente raggiunto quel passo; per
questo le curve delle iterazioni avanzate cambiano composizione e vanno lette
insieme al relativo numero di dataset, non come una serie longitudinale su un
campione costante.

### Osservazioni sulla discovery e priorità sperimentali

La discovery è implementata, ma non è stata eseguita nell'esperimento corrente.
Non è quindi ancora possibile concludere che migliori la recall, che introduca
troppi falsi positivi o che il voting ne aumenti l'affidabilità. Le osservazioni
seguenti derivano dal codice e definiscono ciò che dovrà essere verificato nei
prossimi esperimenti; non sono risultati già osservati.

#### Comportamento già verificabile dal codice

- il critic di discovery viene chiamato soltanto dopo la conferma dei branch
  esistenti e riceve la descrizione del progetto, gli HLG correnti, gli attori
  noti e gli HLG ritirati;
- gli attori noti derivano dagli HLG disponibili: la discovery corrente può
  aggiungere nuovi HLG per questi attori, ma non esegue una nuova estrazione e
  non introduce attori assenti;
- il critic restituisce richieste di generazione, non goal definitivi. Ogni
  richiesta accettata passa dai generatori bottom-up di HLG e LLG e il nuovo
  branch viene rivalutato nelle iterazioni successive;
- una discovery positiva annulla le conferme precedenti, mentre la convergenza
  con discovery attiva richiede anche `NO_MISSING_HIGH_LEVEL_GOALS`;
- il giudizio di copertura è prodotto dal modello sulla descrizione consolidata:
  non esiste ancora una mappa deterministica tra evidenze, attori e goal;
- la ground truth non viene mostrata al critic di discovery e resta confinata al
  notebook di valutazione.

Queste proprietà indicano che il primo passo non dovrebbe essere sostituire
subito la discovery corrente con retrieval, grafi o altri meccanismi più
complessi. Occorre prima misurarne il comportamento sulla stessa baseline e
ground truth usate per le configurazioni senza discovery.

#### Domande da verificare nei prossimi run

Per ciascuna configurazione dei voter, il confronto
`ENABLE_HLG_DISCOVERY=False` contro `True` dovrà stabilire:

1. quanti HLG mancanti della ground truth vengono effettivamente recuperati;
2. quanti HLG aggiunti sono supportati dalla documentazione ma non annotati
   nella ground truth;
3. quanti candidati sono non supportati, duplicati o troppo simili a branch già
   presenti;
4. se l'aumento eventuale della recall compensa la variazione di precisione e
   F1, separatamente per HLG e LLG;
5. se i nuovi branch restano stabili dopo il successivo audit oppure vengono
   riscritti o rimossi;
6. come cambiano convergenza, numero di iterazioni, warning e costo in chiamate;
7. quanto concordano i voter non soltanto sulla decisione finale, ma anche sul
   gap individuato e sulle evidenze che lo giustificano.

La sola decisione `NO_MISSING_HIGH_LEVEL_GOALS` non deve essere interpretata
come prova di completezza. Analogamente, un aumento del numero di goal o della
recall non dimostra da solo che la discovery abbia funzionato: le aggiunte
devono essere controllate rispetto alla documentazione e alla ground truth, e
devono essere distinte dalle normali riscritture o sostituzioni di branch.

#### Ensemble eterogeneo corrente

L'implementazione corrente usa tre voter diversi — Ministral, Bonsai e Gemma —
con lo stesso prompting. HLG e discovery usano il quorum 2/3; la valutazione LLG
applica inoltre il veto di riparazione motivato quando tutti i voti sono validi.
Il model id associato a ogni decisione è riportato nella motivazione aggregata e
la configurazione completa è salvata negli artefatti. La discovery resta un
prompt separato, ma viene valutata dallo stesso ensemble.

#### Miglioramenti successivi, guidati dai risultati

Gli interventi architetturali verranno considerati dopo il confronto
sperimentale, in funzione degli errori osservati:

| Evidenza sperimentale | Possibile intervento successivo |
| --- | --- |
| HLG mancanti associati ad attori assenti | aggiungere una fase separata di discovery degli attori |
| Goal nuovi non supportati o difficili da verificare | introdurre un registro evidence-to-goal con riferimenti alle fonti |
| Duplicati o gap formulati in modo troppo generico | generare più candidati e applicare deduplicazione e validazione esplicite |
| Perdita di capacità durante rewrite o rigenerazione | applicare correzioni additive e un controllo di non regressione prima/dopo |
| Descrizioni lunghe o copertura globale insufficiente | valutare retrieval mirato o aggregazione gerarchica delle evidenze |
| Elevato disaccordo tra voter | analizzare separatamente effetto del modello, prompting e ordine delle evidenze |

Questa priorità mantiene separati due obiettivi: prima misurare l'effetto della
discovery e della composizione dei voter; soltanto dopo modificare il modo in cui
le evidenze vengono recuperate, aggregate o trasformate in nuovi goal. Il piano
completo delle sei condizioni è riportato nella sezione
[Piano per i prossimi esperimenti](#piano-per-i-prossimi-esperimenti).

## Installazione

### Prerequisiti

Assicurarsi che Python e le dipendenze necessarie siano installati.

### Passaggi

1. Clonare il repository:

   ```bash
   git clone https://github.com/dadoluca/LLM_MultiAgentChain_GoalOrientedAPIAlignement.git
   cd LLM_MultiAgentChain_GoalOrientedAPIAlignement
   ```

2. Installare le dipendenze richieste:

   ```bash
   pip install -r requirements.txt
   ```

3. In LM Studio caricare `mistralai/ministral-3-14b-reasoning`,
   `prism-ml/bonsai-27b` e `google/gemma-4-12b-qat`, quindi avviare il server
   locale OpenAI-compatible (default `http://127.0.0.1:1234/v1`).
4. Eseguire `notebook/01_pipeline_execution_top_down_only.ipynb`.
5. Eseguire `notebook/01_pipeline_execution_bottom_up_only.ipynb`.
6. Facoltativamente, eseguire
   `notebook/02_experimental_evaluation_top_down_vs_bottom_up.ipynb`.

Le impostazioni dei modelli possono essere personalizzate tramite variabili
d'ambiente:

| Variabile | Default | Uso |
|---|---|---|
| `QWEN_GENERATOR_MODEL` | `qwen/qwen3.8-27b` | Model id Qwen su Groq |
| `QWEN_GENERATOR_MAX_TOKENS` | `4000` | Budget massimo delle generazioni strutturate |
| `QWEN_RATE_LIMIT_CYCLES` | `3` | Giri massimi delle chiavi in caso di rate limit Groq |
| `QWEN_RATE_LIMIT_WAIT_SECONDS` | `60` | Attesa dopo un giro completo di chiavi temporaneamente limitate |
| `LM_STUDIO_BASE_URL` | `http://127.0.0.1:1234/v1` | Endpoint locale OpenAI-compatible |
| `LM_STUDIO_MANAGEMENT_BASE_URL` | `http://127.0.0.1:1234/api/v1` | Endpoint nativo per caricare e scaricare i modelli |
| `LM_STUDIO_MODEL_MANAGEMENT_TIMEOUT_SECONDS` | `600` | Timeout di load/unload dei modelli locali |
| `LM_STUDIO_EVALUATOR_CONTEXT_LENGTH` | `8192` | Context length usata quando viene caricata un'istanza critic |
| `MINISTRAL_CRITIC_MODEL` | `mistralai/ministral-3-14b-reasoning` | Model id esatto mostrato da LM Studio |
| `BONSAI_CRITIC_MODEL` | `prism-ml/bonsai-27b` | Secondo voter del critic HLG/LLG bottom-up |
| `GEMMA_CRITIC_MODEL` | `google/gemma-4-12b-qat` | Terzo voter del critic HLG/LLG bottom-up |
| `MINISTRAL_CRITIC_MAX_TOKENS` | `700` | Budget massimo di ogni valutazione |
| `DESCRIPTION_SOURCE_MAX_CHARS` | `18000` | Limite testa/coda del README inviato a Qwen per generare la descrizione |

## Licenza

Questo progetto è un fork di
https://github.com/dadoluca/LLM_MultiAgentChain_GoalOrientedAPIAlignement.
È distribuito con licenza **GNU GPL versione 3**. Consultare
[LICENSE](LICENSE) per i dettagli.

## Configurazione adottata per la pipeline

Gli esperimenti di questo repository utilizzano la configurazione
dell'architettura completa con le migliori prestazioni riportate nello studio di
riferimento: il flusso di esecuzione top-down originale con le seguenti
impostazioni specifiche per fase:

| Fase | Prompting del generatore | Critic | Esecuzione |
|---|---|---|---|
| Attori | Zero-shot | Few-shot, abilitato | Ciclo generator-critic |
| High-Level Goal | Zero-shot | Few-shot, abilitato | Ciclo generator-critic |
| Low-Level Goal | Zero-shot | Few-shot, abilitato | Ciclo generator-critic su tutti gli HLG |

Il critic top-down utilizza quattro esempi di valutazione few-shot, una soglia di
qualità pari a `8.5/10` e un massimo di tre iterazioni generator-critic. Questi
parametri sono limitati al notebook top-down e non modificano i valori
predefiniti del bottom-up.

I modelli originali sono stati sostituiti perché deprecati o non più adeguati
all'ambiente di esecuzione corrente. La pipeline utilizza ora:

- **`qwen/qwen3.8-27b` tramite Groq** come generatore di output strutturato;
- **Ministral locale tramite LM Studio** come critic.

L'estrazione di attori, HLG e LLG parte da una generazione zero-shot e utilizza
poi il critic few-shot per assegnare un punteggio al risultato e fornire feedback
per un nuovo tentativo quando la soglia di qualità non viene raggiunta. Generare
e valutare tutti gli LLG a partire dall'insieme completo degli HLG preserva la
semantica della pipeline originale ed evita che ciascun singolo branch venga
valutato erroneamente come se dovesse rappresentare l'intero sistema software.

## Piano per i prossimi esperimenti

La configurazione bottom-up corrente usa un ensemble eterogeneo fisso composto
da Ministral, Bonsai e Gemma. I tre modelli ricevono lo stesso input e le
decisioni HLG usano un quorum 2/3, mentre quelle LLG includono il veto di
rigenerazione motivato. La discovery è un controllo separato eseguito dagli
stessi tre modelli con quorum 2/3.

La stessa baseline dovrà essere eseguita nelle due condizioni seguenti:

| Ensemble HLG/LLG | Discovery disabilitata | Discovery abilitata |
| --- | --- | --- |
| Ministral + Bonsai + Gemma, politica severa documentata sopra | validazione/correzione | validazione/correzione + discovery ensemble |

Per rendere il confronto interpretabile dovranno restare fissi ground truth,
baseline top-down, dataset, prompt condivisi, limiti di iterazione, soglie e
parametri di generazione; tra le due condizioni dovrà cambiare soltanto lo stato
della discovery. Oltre a Precision, Recall e F1 di HLG e LLG,
sarà utile registrare il numero di modifiche, la convergenza, i warning, il
disaccordo tra voter e il costo in chiamate. Nei run con discovery attiva
andranno misurati separatamente anche gli HLG realmente nuovi e supportati, i
duplicati introdotti, le aggiunte non supportate e il recupero dei goal mancanti.
I risultati già presenti nel repository precedono questa modifica e
rappresentano la precedente configurazione omogenea; non devono essere attribuiti
all'ensemble Ministral–Bonsai–Gemma senza rieseguire i notebook.
