# Pipeline bottom-up: comportamento effettivo

Questa pagina descrive il codice corrente di [goal_cycle_orchestrator.py](./goal_cycle_orchestrator.py), [global_goal_evaluator.py](./global_goal_evaluator.py), [goal_reconstructor.py](./goal_reconstructor.py) e del [notebook bottom-up](../../notebook/01_pipeline_execution_bottom_up_only.ipynb).

La pipeline valida e corregge una gerarchia già generata top-down. Un branch è un HLG con i suoi LLG. La ricostruzione produce un HLG diagnostico, chiamato HLG′, per capire se l'incoerenza riguarda il parent o la decomposizione. Il risultato bottom-up viene salvato separatamente dalla baseline.

Il notebook baseline usa la logica originale di `src.utils` e la reflection
originale contenuta in `src.self_critique.refine_response`. Nello stesso modulo,
le funzioni con prefisso `BottomUp`/`bottom_up_` contengono la reflection
focalizzata usata esclusivamente dalle riparazioni. Analogamente,
`src.extraction.extractor` conserva i generatori originali e contiene
`generate_bottom_up_low_level_goals` per il contesto di riparazione. I percorsi
sono distinti tramite nomi espliciti, mentre i client Qwen/Ministral restano
condivisi.

## 1. Ingresso e costruzione dei branch

Il notebook seleziona i file `*_top_down.json` in `output/top_down/` il cui nome di dataset compare in `ALL_GROUNDTRUTHS`. `BASELINE_FILE_OVERRIDES` permette di scegliere file specifici. La ground truth serve qui per selezionare i dataset; non viene fornita ai critic come risposta attesa.

L'esecuzione dei dataset è sequenziale e organizzata in gruppi da 3. Il notebook attende 60 secondi tra due dataset dello stesso gruppo e 120 secondi dopo ogni terzo dataset; la pausa da 120 secondi sostituisce quella da 60 al confine del gruppo. Le pause vengono applicate anche quando un dataset fallisce, tranne dopo l'ultimo input.

Ogni baseline deve contenere `description`, `structuredHighLevelGoals` e `structuredLowLevelGoals`. Non vengono rieseguite l'estrazione iniziale degli attori o dei goal. Le configurazioni della baseline sono conservate come provenienza; le nuove generazioni usano la configurazione bottom-up del notebook.

Il codice crea un branch per ogni HLG, anche senza figli, e associa un LLG al parent contenuto nel suo campo `high_level_associated`. L'associazione richiede uguaglianza strutturale dell'intero oggetto HLG: non è un matching semantico per nome. Un parent assente dalla lista ufficiale interrompe il ciclo con un errore.

Il `branch_id` deriva da un hash dell'HLG completo e da un indice di occorrenza. È un riferimento tecnico per operazioni e contatori, non una misura di similarità.

## 2. Una iterazione è un pass sui branch attivi

Il flusso effettivo è:

1. Costruzione dei branch sullo stato corrente.
2. Ricostruzione sequenziale di HLG′ per tutti i branch attivi.
3. Valutazione sequenziale dei branch: prima Critic 1, poi eventualmente Critic 2 per lo stesso branch.
4. Aggiornamento delle conferme; eventuale Critic 3 solo se la discovery è abilitata e tutti i branch sono confermati.
5. Selezione delle correzioni secondo la priorità descritta sotto e applicazione allo stato.
6. Memorizzazione del riepilogo e salvataggio del trace.
7. Controllo di convergenza/arresto o avvio dell'iterazione successiva.

Nell'iterazione 1 tutti i branch sono attivi. Le modifiche ai goal vengono applicate solo dopo aver raccolto tutti i verdetti del pass: il primo branch non modifica la gerarchia vista dall'ultimo. La memoria del critic, invece, si aggiorna dopo ogni verdetto aggregato e può influenzare i branch successivi dello stesso pass.

I branch già confermati vengono saltati nelle iterazioni successive finché la struttura HLG resta invariata. Dopo rewrite, rimozione o aggiunta HLG, tutte le conferme vengono cancellate. Dopo una correzione solo LLG, il branch da correggere resta attivo; gli altri mantengono le loro conferme.

## 3. Ricostruzione cieca di HLG′

Il reconstructor riceve soltanto le descrizioni degli LLG del branch con etichette locali `llg_NNN`. Non vede il parent originale, gli altri branch o la descrizione del progetto. Inferisce un'intenzione comune e una motivazione tramite Qwen. La ricostruzione deve essere **lossless**: deve rendere visibile ogni capacità distinta dei figli, inclusi eventuali outlier, senza nasconderli dietro verbi generici come *manage*, *handle* o *process*. La motivazione deve associare ogni `llg_NNN` alla capacità che introduce. In questo modo un figlio fuori scope rende HLG′ più ampio o diverso e attiva il routing verso Critic 2.

Il risultato contiene `reconstructed_high_level_goal`, `rationale`, `branch_id` e `source_low_level_goal_ids`. Quest'ultimo campo salva attualmente i nomi degli LLG, non le etichette locali del prompt.

Per un branch vuoto non viene chiamato Qwen: il codice restituisce una ricostruzione esplicita di impossibilità. La mancanza di figli non dimostra che il parent sia errato.

## 4. Critic 1: validazione HLG e diagnosi semantica

Il critic HLG riceve:

- descrizione completa del progetto;
- parent originale con nome, descrizione e attore;
- HLG′ e motivazione del reconstructor, trattata come ipotesi;
- nomi e descrizioni degli LLG del branch corrente;
- ID dei sibling dello stesso attore, loro HLG con nome/descrizione e nomi dei loro LLG.

I sibling sono compattati: le descrizioni dei loro LLG e quelle dei loro attori non vengono inviate. I branch di altri attori sono esclusi perché non possono essere target di `FULLY_REDUNDANT` né ricevere LLG spostati dal branch corrente. Questo riduce il prompt senza indebolire il controllo di consolidamento consentito.

Il prompt richiede prima una validazione indipendente del parent: supporto documentale, autorità dell'attore, risultati essenziali, perimetro e livello di astrazione. Poi confronta HLG e HLG′ e, nei casi previsti, HLG′ con i sibling. Sono giudizi dell'LLM, non confronti deterministici tra stringhe o soglie di embedding.

`EQUIVALENT` richiede implicazione semantica in entrambe le direzioni: stesso attore, stessi esiti essenziali e stesso perimetro. Stesso tema o stessa intenzione principale non bastano. Una copertura incompleta o l'aggiunta di un esito indipendente con sovrapposizione significativa richiede `PARTIAL_OVERLAP`. Un HLG′ generico non prova l'equivalenza: il critic deve enumerare gli esiti materiali delle due formulazioni e le due liste devono coincidere. Il critic legge insieme nome, descrizione e attore e applica la granularità tipica del ground truth: un HLG deve esprimere un risultato funzionale riconoscibile e delimitato. `KEEP + EQUIVALENT` è ammesso soltanto con score 5. L'audit atomico di supporto e completezza dei singoli LLG resta responsabilità esclusiva di Critic 2.

### Matrice di decisione e routing

| Situazione | Decisione HLG | Critic 2 | Azione |
|---|---|---|---|
| HLG′ equivalente, parent documentato e valido | KEEP | No | Conferma branch |
| HLG′ parziale o più ampio, parent valido | KEEP | Sì | Verifica decomposizione |
| HLG′ parziale, parent materialmente errato | REWRITE | No | Riscrive HLG e rigenera LLG |
| HLG′ coincide con un sibling dello stesso attore, parent interamente ridondante | REMOVE | No | Rimuove e richiede consolidamento |
| HLG′ coincide con un sibling, parent distinto e valido | KEEP | Sì | Verifica responsabilità degli LLG |
| HLG′ diverso, parent valido | KEEP | Sì | Verifica decomposizione |
| Parent errato ma recuperabile, anche con HLG′ equivalente | REWRITE | No | Riscrive HLG e rigenera LLG |
| Intenzione completamente non documentata | REMOVE | No | Elimina branch |
| Nessun LLG, parent valido | KEEP | Sì | Richiede decomposizione mancante |

Un HLG e un HLG′ possono essere equivalenti e condividere un errore di scope. Per esempio, entrambi descrivono soltanto l'inserimento dati mentre l'unico esito documentato è completare e inviare una domanda, senza sibling che coprano il resto. Il prompt richiede `REWRITE` del parent incompleto. Se il parent già rappresenta tutto l'esito e soltanto i figli sono incompleti, richiede `KEEP` e Critic 2.

Il prompt operativo contiene 9 esempi compatti strutturati nello stile degli HLG del ground truth, ma non usa le risposte target dei dataset. Copre equivalenza autentica, copertura parziale, attore scorretto, parent incompleto, sibling con parent distinto, ridondanza completa, divergenza e intenzione non documentata. La motivazione deve indicare evidenza e differenze funzionali concrete; difetti puramente stilistici non giustificano rewrite.

### Vincoli Python sul risultato

Il modello [HighLevelGoalEvaluation](../data_model.py) salva decisione, score, relazione, flag di routing e metadati della correzione. Pydantic impone:

- `REWRITE` richiede `rewriting_focus`;
- `REMOVE` richiede `removal_basis`;
- `FULLY_REDUNDANT` richiede `MATCHES_SIBLING` e ID/nome del branch coprente;
- `UNSUPPORTED` non deve indicare un branch coprente;
- `KEEP + EQUIVALENT` impone `requires_llg_evaluation=false`;
- `KEEP + EQUIVALENT` richiede `quality_score=5`;
- `KEEP` con altra relazione impone `requires_llg_evaluation=true`;
- `REWRITE` e `REMOVE` impongono il flag a false.

Il codice verifica inoltre che `NOT_RECONSTRUCTABLE` corrisponda effettivamente a un branch vuoto. Per una rimozione ridondante controlla ID, nome HLG e nome dell'attore del sibling. Se la base di rimozione non supera questi controlli, il verdetto viene convertito in KEEP con motivazione del blocco e score 4; il routing dipende dalla relazione.

## 5. Critic 2: verifica della decomposizione

`evaluate_branch()` invoca il critic LLG per ogni HLG mantenuto. Anche un HLG′ equivalente dimostra soltanto che il significato aggregato dei figli coincide con il parent: non certifica supporto, necessità e non-ridondanza di ciascun LLG. Il critic riceve progetto, parent, HLG′ e motivazione, LLG correnti con ID locali, nomi e descrizioni, e HLG sibling compattati.

Le decisioni sono `KEEP_LOW_LEVEL_GOALS` e `REGENERATE_LOW_LEVEL_GOALS`. Il prompt cerca capacità essenziali mancanti, figli non documentati o fuori scope, duplicazioni, frammenti puramente UI/tecnici e responsabilità di sibling. Non richiede rigenerazione solo per stile o per l'assenza di funzionalità opzionali. Un verbo ampio nel parent non autorizza automaticamente tutte le operazioni convenzionali sullo stesso oggetto.

`KEEP_LOW_LEVEL_GOALS` richiede ora score 5, liste dei difetti vuote e `regeneration_feedback=null`. La motivazione deve mappare ogni `llg_NNN` a un requisito documentato o dimostrarne la necessità stretta e deve coprire tutti gli esiti essenziali del parent. `REGENERATE_LOW_LEVEL_GOALS` richiede invece feedback operativo e almeno un ID problematico valido o una capacità essenziale mancante.

Dopo il voting, il codice elimina gli ID problematici inesistenti e le descrizioni vuote di capacità mancanti. Una richiesta REGENERATE senza almeno un ID valido o una capacità mancante non vuota viene rifiutata: non può più essere convertita automaticamente in KEEP. Il branch diventa inconcludente e potrà essere rivalutato, evitando che un errore formale del critic produca una conferma permissiva.

Per un branch vuoto, un KEEP del critic LLG viene forzato a REGENERATE. La conferma senza decomposizione è quindi impedita dal normale percorso di validazione.

## 6. Voting: tre modelli con lo stesso input

I critic HLG e LLG usano un ensemble eterogeneo fisso: `mistralai/ministral-3-14b-reasoning`, `prism-ml/bonsai-27b` e `google/gemma-4-12b-qat`. I tre modelli ricevono lo stesso system prompt, lo stesso prompt utente per il medesimo branch e lo stesso snapshot della memoria. L'ordine è model-major: dopo che Qwen ha ricostruito i branch uno alla volta, LM Studio carica Ministral una sola volta e gli fa valutare in sequenza tutti i branch HLG; lo scarica e ripete l'intero insieme con Bonsai, poi con Gemma. Gli HLG vengono aggregati con quorum 2/3. Ogni HLG mantenuto passa poi allo stadio LLG, che usa lo stesso batching per modello. Se tutti e tre i voti LLG sono validi, un singolo `REGENERATE` con prove strutturate attiva la riparazione; con un pannello incompleto torna ad applicarsi il quorum ordinario. In questo modo ogni stadio richiede al massimo tre load/unload, non tre cambi di modello per ogni branch. Anche REMOVE usa il quorum ordinario: non serve unanimità e restano attivi i gate distruttivi. La discovery è un giudizio separato, ma usa lo stesso ensemble e lo stesso quorum 2/3; il suo prompt aggiunge esplicitamente la condizione di ricerca degli HLG mancanti.

Tutti i prompt di giudizio sono few-shot. Il critic HLG contiene 9 esempi di calibrazione, il critic LLG ne contiene 7 e il prompt separato di discovery ne contiene 3. Ogni membro dell'ensemble riceve integralmente gli stessi esempi insieme allo stesso caso da valutare. Il critic Ministral usato nella reflection delle correzioni Qwen usa invece i 4 esempi few-shot configurati dal notebook.

Il critic LLG riceve anche il verdetto HLG aggregato. Se il quorum HLG ha classificato la ricostruzione come `PARTIAL_OVERLAP`, `DIFFERENT`, `MATCHES_SIBLING` o `NOT_RECONSTRUCTABLE`, il branch non può essere confermato invariato da un successivo `KEEP_LOW_LEVEL_GOALS`: il prompt deve localizzare la causa del mismatch e un controllo deterministico converte comunque un eventuale KEEP contraddittorio in una rigenerazione mirata. Anche `EQUIVALENT` passa ora all'audit LLG indipendente. Un voto severo costituisce veto soltanto se è un `REGENERATE` valido, contiene almeno una prova strutturata e tutti e tre i voter hanno risposto validamente.

Ogni voter ha fino a 3 tentativi complessivi. Le risposte devono essere JSON validi, conformi a Pydantic e coerenti con lo score:

| Score | Significato | Decisioni ammesse |
|---|---|---|
| 0–3 | Difetto materiale o residuo che richiede correzione | REWRITE, REMOVE, REGENERATE, FOUND |
| 4–5 | Artefatto supportato e senza difetti materiali | KEEP HLG, NO_MISSING |

Per Critic 2 vale un vincolo più severo: `KEEP_LOW_LEVEL_GOALS` richiede esattamente 5/5; uno score 3 o 4 non è sufficiente a confermare la decomposizione.

Il quorum riguarda la decisione (`KEEP`, `REWRITE`, `REMOVE`, ecc.). Per Critic 1, quando almeno 2 voter scelgono `KEEP` ma divergono sulla relazione, viene conservato un singolo voto coerente appartenente alla maggioranza e viene preferito quello più rigoroso. Tutti gli HLG mantenuti passano comunque a Critic 2. `REMOVE` e `REWRITE` richiedono almeno 2 voti su 3 sulla rispettiva azione; soltanto `REGENERATE_LOW_LEVEL_GOALS` dispone del veto motivato descritto sopra.

Il risultato conserva un voto rappresentativo del gruppo vincente, non la media degli score e non l'unione delle evidenze. Per un `KEEP` HLG viene privilegiato il routing verso Critic 2; negli altri casi la scelta privilegia il voto con più elementi nelle liste di problemi/gap. I metadati di rimozione del voto rappresentativo superano poi i controlli deterministici sul sibling coprente.

Nel trace si salvano `voter_count`, `valid_vote_count`, `winning_vote_count` e `vote_distribution`. La motivazione aggregata registra inoltre la decisione restituita da ciascun model id; il contenuto completo dei singoli voti non viene persistito.

## 7. Applicazione delle correzioni

La priorità è esclusiva tra categorie, ma applica insieme tutte le correzioni indipendenti della categoria selezionata:

1. Tutti i rewrite HLG validi del pass. Per ciascun branch viene accettato al massimo un parent generato e viene rigenerata separatamente la sua decomposizione.
2. Se non è stato applicato alcun rewrite, tutte le rimozioni indipendenti e sicure. I target di consolidamento sono protetti nello stesso batch per evitare rimozioni circolari.
3. Se non sono stati applicati rewrite né rimozioni, tutte le rigenerazioni LLG richieste dal pass, una chiamata di generazione separata per branch.

La separazione delle chiamate LLG evita che un batch grande superi i limiti di contesto e impedisce che un errore su un branch annulli le altre correzioni. Una rimozione `UNSUPPORTED` elimina parent e figli. Una rimozione `FULLY_REDUNDANT` prepara la rigenerazione del target usando i candidati LLG di entrambi i branch.

Le nuove generazioni riusano Qwen e l'infrastruttura condivisa, ma passano dalla reflection focalizzata bottom-up `generate_bottom_up_response_with_reflection`; non modificano né invocano il percorso di reflection della baseline top-down. Nel notebook: Zero-shot, critic few-shot attivo con 4 esempi, soglia 8/10 e fino a 5 tentativi. Questo critic generativo è distinto dal voting bottom-up e non usa automaticamente tre voti. Il parametro `evaluator_ablation` controlla questa reflection, non disattiva Critic 1 o Critic 2.

La reflection può restituire l'ultimo risultato anche sotto soglia dopo l'esaurimento dei tentativi: la soglia non è una garanzia di accettazione assoluta. Il ciclo successivo valida di nuovo i branch.

Gli LLG generati devono preservare esattamente il parent atteso; quelli senza un'associazione univoca vengono scartati con warning. La deduplicazione usa parent e nome/descrizione normalizzati: non è deduplicazione semantica. Un insieme non vuoto sostituisce l'intera decomposizione, senza correzione selettiva. Se la generazione LLG fallisce, è vuota oppure tutti i risultati vengono scartati per parent non identico, il codice conserva i candidati precedenti (riassociandoli al parent riscritto quando necessario) e registra il warning corrispondente. Un errore non gestito nella generazione del nuovo HLG, invece, interrompe il dataset e viene registrato dal notebook.

La preservazione delle capacità valide viene richiesta al generatore, ma non verificata automaticamente mediante un controllo di non regressione.

Il critic preferisce `REMOVE` a `REWRITE` quando un HLG non lascia alcuna intenzione documentata autonoma dopo la rimozione delle parti errate, oppure quando è un frammento di workflow interamente assorbito da un preciso HLG sibling dello stesso attore. Restano obbligatori il quorum, l'identificazione esatta del sibling per `FULLY_REDUNDANT`, il dependency guard e il limite di due rimozioni per iterazione.

## 8. Due output: senza e con discovery

Per ogni baseline il notebook avvia due cicli indipendenti dallo stesso stato top-down: prima senza discovery e poi con discovery. Il primo salva in `output/top_down_bottom_up/`; il secondo salva in `output/top_down_bottom_up_discovery/`. Anche i trace delle iterazioni sono separati, rispettivamente in `output/bottom_up_iterations/` e `output/bottom_up_iterations_discovery/`, quindi un profilo non sovrascrive l'altro.

Nel secondo profilo la discovery parte solo quando tutte le conferme dei branch sono disponibili e non precede le riparazioni. La valutano Ministral, Bonsai e Gemma con lo stesso input, gli stessi esempi few-shot e quorum 2/3. Rispetto ai prompt ordinari, il prompt discovery contiene la condizione esplicita di verificare l'intera descrizione e individuare almeno un'intenzione funzionale actor-level documentata ma assente dagli HLG correnti.

Il critic riceve progetto, HLG correnti, attori noti e HLG ritirati; restituisce richieste focalizzate, non goal definitivi. Gli attori noti derivano dagli HLG iniziali e successivi, non da un nuovo pass di estrazione. Non vengono introdotti attori nuovi.

Ogni richiesta valida genera al massimo un HLG e poi la sua decomposizione indipendente. Non c'è un tetto complessivo di nuovi HLG per pass; restano i limiti di token. Le nuove aggiunte cancellano le conferme e passano alla validazione successiva. Un nuovo HLG senza figli rimane nel ciclo come branch vuoto.

## 9. Memoria e token

La memoria è isolata per progetto e condivisa dai critic bottom-up attivi. Conserva al massimo 4 scambi aggregati e 2 riepiloghi di stato. La richiesta include solo l'ultimo riepilogo compatto e gli scambi recenti che rientrano nel budget di 1.000 caratteri. Il riepilogo contiene soltanto contatori di HLG/LLG aggiunti, rimossi o modificati, non elenchi completi.

I voter della stessa valutazione vedono lo stesso stato della memoria; nessuno vede il voto degli altri. Dopo l'aggregazione viene registrato il risultato. Ricostruzione e generazione non condividono questa memoria.

Il modello generatore ha default `qwen/qwen3.8-27b` su Groq, modificabile con `QWEN_GENERATOR_MODEL`. I tre model id dell'ensemble locale sono configurabili con `MINISTRAL_CRITIC_MODEL`, `BONSAI_CRITIC_MODEL` e `GEMMA_CRITIC_MODEL`; sono usati per HLG, LLG e discovery. La reflection delle rigenerazioni usa invece il solo `MINISTRAL_CRITIC_MODEL`. I voter bottom-up usano l'endpoint nativo LM Studio con un budget predefinito di 1.600 token: Ministral mantiene il reasoning, mentre per Bonsai e Gemma viene disattivato per garantire l'emissione del JSON finale. Pydantic verifica campi, enum e coerenza decisione/score e, se parsing o validazione falliscono, viene ripetuto soltanto il voto del modello interessato. Il system prompt limita inoltre la motivazione a 200 parole per ridurre il rischio di troncamento.

## 10. Convergenza, arresti ed errori

Il ciclo converge quando tutte le conferme sono disponibili e nel pass non sono applicati rewrite, rimozioni o aggiunte. Con discovery abilitata serve anche `NO_MISSING_HIGH_LEVEL_GOALS`. La prima iterazione può bastare se tutti i branch superano il percorso previsto.

`converged=True` esprime accordo dei critic sulla gerarchia; non dimostra correttezza rispetto alla ground truth e non implica che tutti i branch siano passati da Critic 2.

| Stop reason restituito | Significato |
|---|---|
| ALL_BRANCHES_CONFIRMED | Gerarchia confermata e discovery soddisfatta se abilitata |
| NO_ACTIONS_REMAIN_WITH_WARNINGS | Non restano azioni riconosciute come eseguibili, ma rimangono warning |
| MAX_ITERATIONS_REACHED | Budget di iterazioni esaurito senza convergenza |

`NO_CHANGES_APPLIED` esiste nell'enum, ma non viene restituito dall'orchestratore corrente.

Fallimenti del critic o assenza di quorum producono normalmente `EVALUATION_INCONCLUSIVE` e consentono un retry nel pass successivo. I fallimenti di dimensione riconosciuti vengono marcati `NON_RETRYABLE_EVALUATOR_REQUEST_TOO_LARGE`; se altre azioni restano disponibili, il ciclo può ancora continuare. La classificazione specifica riguarda i branch, non applica la stessa distinzione agli errori della discovery.

Errori non gestiti, inclusi quelli dei generatori, interrompono il dataset: il notebook registra `bottomUpError`, chiude il client e prosegue con il dataset seguente. Questo è diverso da un risultato regolare con `converged=False`.

## 11. Configurazione e limiti effettivi

| Parametro | Default/corrente | Ambito |
|---|---|---|
| DEFAULT_GLOBAL_CYCLE_MAX_ITERATIONS | 5 | Libreria |
| GLOBAL_CYCLE_MAX_ITERATIONS_OVERRIDE | 25 | Notebook |
| DISCOVERY_RUN_MODES | (False, True) | Due profili indipendenti per ogni baseline |
| MAX_HLG_REWRITES_PER_BRANCH | 2 | Contatore per ID HLG |
| MAX_HLG_REWRITES_PER_ITERATION | 2 | Rewrite HLG applicati nello stesso pass; gli altri vengono rivalutati nel pass successivo |
| MAX_LLG_REGENERATIONS_PER_BRANCH | 2 | Correzioni LLG ordinarie per ID |
| MAX_LLG_REGENERATIONS_PER_ITERATION | 2 | Rigenerazioni LLG applicate nello stesso pass; le altre vengono rivalutate nel pass successivo |
| MAX_HLG_REMOVALS_PER_ITERATION | 2 | Rimozioni HLG applicate nello stesso pass, dopo il dependency guard |
| MAX_REPLACEMENT_HLGS_PER_REWRITE | 1 | Parent accettati per rewrite |
| BOTTOM_UP_EVALUATOR_VOTERS | 3 | Configurabile: 3 o 5 |
| EVALUATOR_VOTING_QUORUM | 2 | Derivato dal numero di voter |
| BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS | 3 | Configurabile: 1–3 |
| BOTTOM_UP_EVALUATOR_TEMPERATURE | 0.2 | Default configurabile |
| BOTTOM_UP_QUALITY_THRESHOLD | 3/5 | Soglia bottom-up |
| Esempi bottom-up | HLG 9, LLG 7, discovery 3 | Prompt separati |
| BOTTOM_UP_CRITIC_MAX_TOKENS | 1600 | Output dei tre voter bottom-up; reasoning disattivato per Bonsai e Gemma |
| REQUEST_TIMEOUT_SECONDS / API_MAX_RETRIES | 120 / 1 | Client Qwen/Groq e Ministral/LM Studio creati dal notebook |

Non esiste un limite fisso al numero di LLG contenuti in una singola rigenerazione. I limiti per iterazione riducono invece il numero di branch modificati contemporaneamente; il limite globale delle iterazioni resta la protezione complessiva contro oscillazioni.

## 12. Trace, file salvati e riesecuzione

I file `iteration_NNN.json` vengono aggiornati dopo ogni decisione di branch. Durante il pass espongono `checkpoint_complete=false` e i contatori `completed_branch_count`/`total_branch_count`; al termine sono sovrascritti con lo stato completo e `checkpoint_complete=true`. Ogni verdetto conserva in `critic_votes` i tre record individuali (modello, risposta grezza, JSON interpretato oppure errore) oltre al risultato aggregato. Il trace contiene inoltre ricostruzioni, gerarchia dopo le correzioni, valutazione discovery o null, contatori LLG e warning. Pertanto un verdetto può riferirsi a un parent precedente che non compare più nella gerarchia finale del medesimo file. I branch congelati non vengono riportati come nuove valutazioni; i contatori di rewrite non sono persistiti nel trace.

Nel risultato del notebook, `globalGoalCycle` contiene l'intero ciclo, mentre `finalHighLevelGoals` e `finalLowLevelGoals` contengono gli artefatti finali. I campi strutturati originari della baseline restano nel JSON combinato: per il risultato corretto bisogna leggere i campi finali.

- `output/top_down/`: baseline invariata.
- `output/bottom_up_iterations/<baseline>/`: trace.
- `output/top_down_bottom_up/`: risultato o artefatto di errore.

A ogni riesecuzione il notebook elimina i precedenti `iteration_*.json` della baseline selezionata e sovrascrive il suo risultato combinato. Per conservare un confronto tra configurazioni occorre archiviare gli output prima del rerun. Dopo modifiche ai moduli, rieseguire la cella con `importlib.reload`; dopo modifiche allo schema Pydantic è preferibile riavviare il kernel.

Precision, Recall e F1 vengono calcolati separatamente nel [notebook di valutazione](../../notebook/02_experimental_evaluation_top_down_vs_bottom_up.ipynb). Le metriche già presenti in `output/evaluation/top_down_vs_bottom_up/` possono riferirsi a configurazioni precedenti: non attestano automaticamente gli effetti dell'ultimo prompt.

## 13. Criticità emerse dall'analisi del codice

Questi limiti descrivono l'implementazione attuale, non correzioni applicate con questa revisione della documentazione.

- **Conferma aggregata:** `EQUIVALENT + KEEP` non basta a confermare il branch; tutti gli LLG vengono comunque controllati. Restano possibili errori semantici condivisi dai tre modelli, da misurare sperimentalmente.
- **Contatore rewrite:** l'ID dipende dal contenuto dell'HLG. Una riscrittura cambia normalmente ID, quindi il limite di 2 non segue tutta la discendenza del branch. Il limite globale resta operativo.
- **Preservazione dopo generazione:** l'accettazione controlla non-vuoto, associazione e duplicati, non la conservazione delle capacità. Un consolidamento può perdere dettagli nonostante il prompt.
- **Duplicati HLG strutturalmente identici:** gli LLG vengono associati al primo parent uguale; gli altri branch possono risultare vuoti. Anche l'associazione di LLG generati a più parent identici può essere ambigua.
- **Contesto delle rigenerazioni:** il generatore LLG riceve scope, feedback e nomi dei candidati già presenti, ma non riceve automaticamente tutta la descrizione originale né tutte le descrizioni dei candidati. Nei rewrite e nei consolidamenti i candidati vengono prima riassociati al parent target, quindi i loro nomi compaiono nei vincoli; la perdita di dettaglio resta possibile perché le descrizioni complete non sono incluse in quella lista sintetica.
- **Voting e copertura:** il quorum garantisce accordo sulla decisione/diagnosi prevista dalla firma, non consenso sulle stesse evidenze. Lo score non è una probabilità di correttezza.
- **Trace e confronto sperimentale:** mancano i voti individuali e una firma/versione del prompt; il rerun sostituisce gli artefatti. Il rafforzamento del prompt deve essere valutato su output nuovi.
- **Verifica automatica:** sono stati eseguiti controlli locali simulati dei percorsi principali, ma nel repository non è presente una suite persistita che copra tutti i casi della matrice e gli errori di generazione.

Per B branch attivi, V voter e L branch instradati verso Critic 2, il pass usa fino a B ricostruzioni (i branch vuoti non chiamano Qwen), V × B voti HLG e V × L voti LLG; con discovery aggiunge V voti di copertura. Retry e reflection possono aumentare le chiamate effettive.
