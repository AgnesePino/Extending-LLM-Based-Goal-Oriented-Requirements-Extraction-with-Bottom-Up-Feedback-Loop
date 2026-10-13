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

## Ciclo bottom-up corrente di validazione e correzione

La baseline usa la logica originale di `src.extraction.extractor`, `src.utils`
e `src.self_critique.refine_response`: prompt dell'allegato, parsing originale,
soglia 8,5 e 3 tentativi. Gemini/Qwen, Zero-shot e critic attivo restano la
configurazione sperimentale corrente.
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

### Obiettivi e perimetro dell'input

La fase bottom-up ha un unico obiettivo principale: **validare e correggere** la
gerarchia prodotta dalla baseline, rimuovendo ridondanze effettive e riparando le
decomposizioni HLG-LLG incoerenti.

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
4. invoca il secondo critic soltanto per un `KEEP` con
   `requires_llg_evaluation=true`. Un `KEEP` equivalente e documentato conferma
   direttamente il branch, ma richiede uno score HLG di 5/5; `REWRITE` e
   `REMOVE` non valutano la vecchia
   decomposizione;
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
aggiornamenti, ma invia solo l'ultimo. Il budget di 2.500 caratteri limita gli
scambi, non il riepilogo o il prompt corrente, anziché riprodurre
integralmente i prompt precedenti.

Ciascun prompt del critic include esempi few-shot specifici per il proprio
compito. Il prompt HLG calibra `KEEP`, `REWRITE` e `REMOVE`; quello LLG confronta
decomposizioni valide e associabili alle API con difetti sostanziali di
copertura. Il prompt globale opzionale distingue le intenzioni degli attori
realmente mancanti dalle varianti più ristrette di obiettivi già coperti.

Ogni decisione del valutatore bottom-up utilizza tre chiamate parallele
indipendenti e una maggioranza di 2 su 3. Per il primo critic il quorum riguarda
anche le rimozioni HLG, senza requisito di unanimità, mantenendo i controlli
sulla base della rimozione e sul sibling coprente. Il voto aggregato considera
la terna completa relazione/decisione/routing, evitando di aggregare campi
provenienti da diagnosi incompatibili. Ogni voter assegna inoltre
all'artefatto corrente un punteggio di qualità da 0 a 5 usando esempi specifici:
i punteggi 3, 4 e 5 sono sufficienti, mentre 0, 1 e 2 richiedono una correzione.
Un voto il cui punteggio contraddice la decisione viene scartato prima
dell'aggregazione. È possibile usare cinque voter con quorum 3 su 5 impostando
`BOTTOM_UP_EVALUATOR_VOTERS=5`. Tutti i voter ricevono lo stesso snapshot
immutabile della conversazione e soltanto il verdetto aggregato viene scritto
nella memoria e nel trace dell'iterazione. Per impostazione predefinita, una
risposta individuale non valida dispone di tre tentativi complessivi (due retry);
`BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS` accetta valori da 1 a 3.

Il routing HLG–LLG segue questo albero decisionale:

| Relazione diagnosticata | Stato del parent | Decisione | Critic LLG |
|---|---|---|---|
| `EQUIVALENT` | documentato e valido | `KEEP` | no |
| `PARTIAL_OVERLAP` | valido | `KEEP` | sì |
| `PARTIAL_OVERLAP` | formulato male | `REWRITE` | no |
| `MATCHES_SIBLING` | interamente ridondante | `REMOVE` | no |
| `MATCHES_SIBLING` | distinto e valido | `KEEP` | sì |
| `DIFFERENT` | valido | `KEEP` | sì |
| qualsiasi, anche `EQUIVALENT` | materialmente errato o incompleto, ma recuperabile | `REWRITE` | no |
| qualsiasi | intenzione completamente non documentata | `REMOVE` | no |
| `NOT_RECONSTRUCTABLE` | valido ma senza LLG | `KEEP` | sì, per generare gli LLG |

Le riscritture degli HLG e le rigenerazioni degli LLG bottom-up utilizzano una
generazione zero-shot seguita dal critic
top-down few-shot abilitato. In questa fase l'estrazione degli attori non viene
ripetuta: vengono riutilizzati gli attori persistiti dalla baseline. La
configurazione di generazione bottom-up è esplicita e non eredita il profilo di
estrazione di una baseline precedente. Le generazioni dovute alla discovery
seguono lo stesso percorso, ma avvengono soltanto quando il modulo è abilitato.

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
| `REWRITE_ORIGINAL_HIGH_LEVEL_GOAL` | Chiama il generatore HLG top-down, sostituisce un HLG e ne rigenera gli LLG | Sostituzione, non discovery netta |
| `REMOVE_ORIGINAL_HIGH_LEVEL_GOAL` | Elimina il branch; se ridondante, richiede consolidamento degli LLG nel target coprente | No; può rigenerare un branch mantenuto |
| `REGENERATE_LOW_LEVEL_GOALS` | Chiama il generatore LLG top-down per lo stesso HLG parent | No |
| `MISSING_HIGH_LEVEL_GOALS_FOUND` | In modalità discovery, chiama il generatore HLG top-down e poi quello LLG top-down | Sì |
| `EVALUATION_INCONCLUSIVE` | Lascia invariata la gerarchia corrente e registra un warning | No |

Le correzioni alla gerarchia degli obiettivi vengono ordinate deliberatamente:
in una singola iterazione vengono applicate al massimo una riscrittura HLG e due
rimozioni HLG, mentre viene rigenerato un solo branch LLG alla volta. La
discovery opzionale degli HLG mancanti non ha un limite numerico quando è
abilitata: ogni gap distinto e documentato restituito dal critic di copertura
viene generato una volta che la gerarchia è stabile. Un branch esistente può essere rigenerato o riscritto al
massimo due volte. Questi limiti sono protezioni ingegneristiche contro cicli di
correzione ripetuti, non soglie ottimizzate empiricamente.

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
LLG top-down creano un nuovo branch. Il processo bottom-up non è quindi limitato
alla rimozione o alla riscrittura della baseline.

Quando viene abilitata, la discovery presenta tuttavia importanti limitazioni:

- la discovery si basa su un'unica descrizione consolidata del progetto, anziché
  su una raccolta indicizzata di passaggi sorgente;
- la discovery è limitata agli attori già rappresentati nell'insieme corrente di
  HLG;
- la ridondanza degli HLG viene decisa esclusivamente dal critic HLG sulla base
  del contesto semantico completo e degli LLG associati; l'orchestratore non
  blocca né fonde HLG perché i rispettivi valori normalizzati `actor::name`
  coincidono;
- gli LLG rigenerati vengono accettati quando non sono vuoti; non è presente un
  controllo semantico deterministico della copertura prima e dopo la modifica;
- gli HLG rimossi e sostituiti vengono nuovamente sottoposti ad audit tramite una
  checklist conservata, ma il giudizio finale sulla copertura rimane basato sul
  modello;
- le tre chiamate specializzate al valutatore aumentano il costo al crescere del
  numero di branch attivi.

La valutazione attualmente salvata mostra una precisione maggiore con una
riduzione più contenuta della recall. La macro F1 degli HLG aumenta da `0.5504`
a `0.6299` e quella degli LLG da `0.3768` a `0.4300`. La recall degli HLG
passa da `0.7507` a `0.7070`, mentre quella degli LLG passa da `0.4888` a
`0.4579`. Il numero di HLG diminuisce da 105 a 70 e quello degli LLG da 457 a
328. Questi file precedono l'audit di discovery senza limite numerico e il prompt
di rigenerazione LLG completo e atomico; devono quindi essere rigenerati prima
di poter essere considerati risultati definitivi per il codice corrente.

Nessuno dei trace attualmente salvati contiene una decisione finale
`MISSING_HIGH_LEVEL_GOALS_FOUND`. Il percorso di discovery è implementato e
testato dal ciclo, ma questo snapshot sperimentale non dimostra ancora il
recupero di un'intenzione funzionale realmente assente. Le misurazioni
dettagliate sono disponibili in
`output/evaluation/top_down_vs_bottom_up/top_down_vs_bottom_up_macro_summary.csv`
e
`output/evaluation/top_down_vs_bottom_up/top_down_vs_bottom_up_deltas.csv`.

### Roadmap di sviluppo orientata alla discovery

Gli elementi seguenti sono miglioramenti proposti e **non sono implementati
nella pipeline corrente**. Sono ordinati in funzione di un obiettivo orientato
alla recall, continuando comunque a proteggere la precisione.

1. **Costruire un registro di copertura evidence-to-goal.** Suddividere ogni
   artefatto sorgente disponibile in passaggi indirizzabili e richiedere che ogni
   attore, HLG e LLG conservi gli ID delle fonti. Estrarre capacità candidate da
   ogni passaggio e mantenere uno stato esplicito `covered`,
   `partially covered`, `uncovered` o `conflicting`. La discovery può quindi
   concentrarsi sulle evidenze non coperte, anziché chiedere una sola volta al
   modello se manca qualcosa. La generazione basata sul retrieval può inoltre
   fornire l'attribuzione delle fonti e ridurre le generazioni non supportate,
   come mostrato in
   [In-Context Retrieval-Augmented Language Models](https://aclanthology.org/2023.tacl-1.75/).

2. **Aggiungere una vera fase di discovery degli attori.** Eseguire l'estrazione
   degli attori sulle evidenze non coperte e consentire a un attore supportato da
   evidenze di entrare in `known_actors` prima della generazione del relativo
   HLG. Richiedere almeno un passaggio sorgente e un'intenzione funzionale
   distinta. In questo modo la discovery supera l'attuale limite degli attori
   già esistenti.

3. **Separare la discovery dalla validazione distruttiva.** Produrre e validare
   prima un insieme additivo di candidati su una baseline immutabile; eseguire
   solo successivamente eventuali fusioni, riscritture e rimozioni. Un nuovo
   candidato non dovrebbe essere soppresso soltanto perché un altro branch viene
   corretto nella stessa iterazione.

4. **Utilizzare un retrieval mirato e iterativo.** Interrogare separatamente le
   evidenze per attore, entità di dominio, workflow, fase del ciclo di vita e
   capacità attualmente non coperta. Reinserire ogni gap individuato nella fase
   di retrieval successiva. Questo approccio segue il pattern
   retrieve-generate-retrieve studiato da
   [Iter-RetGen](https://aclanthology.org/2023.findings-emnlp.620/) ed evita di
   dipendere da un unico prompt globale.

5. **Utilizzare un'aggregazione gerarchica e globale delle evidenze.** La
   discovery dei requisiti è un problema di comprensione dell'intero corpus, non
   soltanto di ricerca del passaggio più vicino. Un grafo o un indice gerarchico
   può collegare attori, risorse, eventi, vincoli e operazioni prima di proporre
   intenzioni mancanti. Lo
   [studio sulla summarization globale di GraphRAG](https://arxiv.org/abs/2404.16130)
   riporta una maggiore completezza e varietà nelle domande globali su corpus di
   grandi dimensioni.

6. **Correggere i branch in modo additivo.** In presenza di un gap LLG,
   conservare gli LLG esistenti supportati e generare soltanto le capacità
   esplicitamente mancanti. Per la rinomina o la correzione dell'ambito di un
   HLG, riassegnare prima il parent degli LLG supportati e poi generare le
   aggiunte. Eseguire un confronto di copertura shadow prima e dopo la modifica e
   rifiutare una variazione che perda una capacità supportata dalle fonti. Questo
   affronta direttamente la regressione di Genome Nexus.

7. **Generare un insieme eterogeneo di candidati e poi validarlo.** Produrre più
   candidati di discovery da prospettive indipendenti basate su attori, workflow
   e risorse dati; raggruppare i duplicati semantici e mantenere i candidati
   supportati da evidenze distinte. È più sicuro che aumentare senza vincoli il
   numero di obiettivi generati. I prompt lunghi, da soli, non sono sufficienti:
   i fatti rilevanti possono essere sottoutilizzati in base alla loro posizione,
   come documentato in
   [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/).

8. **Rafforzare le decisioni distruttive.** Richiedere due giudizi indipendenti o
   un controllo deterministico delle evidenze prima di `REMOVE` o di un
   `REWRITE` con perdita di informazioni. Randomizzare l'ordine dei candidati
   nei confronti a coppie e riportare i disaccordi. I giudici LLM possono
   manifestare bias di posizione, verbosità e auto-preferenza, come documentato
   da
   [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://papers.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html).

9. **Terminare in base alla saturazione misurata della copertura.** Interrompere
   la discovery soltanto dopo aver controllato tutte le unità di evidenza e
   quando più passaggi indipendenti non aggiungono nuove capacità validate.
   Mantenere contatori separati per branch scoperti, fusi, riscritti e rimossi;
   `NO_MISSING_HIGH_LEVEL_GOALS` restituito da una singola chiamata al modello
   non dovrebbe essere l'unico criterio di completezza.

10. **Valutare direttamente la discovery.** Oltre a precisione, recall e F1,
    riportare la copertura delle evidenze, il numero di HLG/LLG realmente nuovi e
    supportati, il tasso di aggiunte non supportate, la perdita distruttiva di
    copertura, la recall degli attori e i risultati di esecuzioni ripetute con
    intervalli di confidenza. Il feedback iterativo è una strategia utile in fase
    di inferenza, come mostrato in
    [Self-Refine](https://openreview.net/pdf?id=S37hOerQLB), ma il ciclo di
    feedback deve ottimizzare esplicitamente l'obiettivo desiderato. Per i gap
    ambigui, è preferibile mantenere una coda di validazione umana anziché
    aggiungere o rimuovere silenziosamente un requisito.

Per questo repository, il primo incremento di implementazione consigliato
comprende gli elementi 1, 2, 3 e 6: copertura a livello di evidenza, discovery di
nuovi attori, fase di discovery additiva e controllo di non regressione. Nel
complesso, questi interventi agiscono direttamente sulla recall senza modificare
la baseline top-down originale.

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

3. Eseguire `notebook/01_pipeline_execution_top_down_only.ipynb`.
4. Eseguire `notebook/01_pipeline_execution_bottom_up_only.ipynb`.
5. Facoltativamente, eseguire
   `notebook/02_experimental_evaluation_top_down_vs_bottom_up.ipynb`.

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

- **Gemini 2.5 Flash** come generatore di output strutturato;
- **Qwen 3.8 27B tramite Groq** come critic.

L'estrazione di attori, HLG e LLG parte da una generazione zero-shot e utilizza
poi il critic few-shot per assegnare un punteggio al risultato e fornire feedback
per un nuovo tentativo quando la soglia di qualità non viene raggiunta. Generare
e valutare tutti gli LLG a partire dall'insieme completo degli HLG preserva la
semantica della pipeline originale ed evita che ciascun singolo branch venga
valutato erroneamente come se dovesse rappresentare l'intero sistema software.
