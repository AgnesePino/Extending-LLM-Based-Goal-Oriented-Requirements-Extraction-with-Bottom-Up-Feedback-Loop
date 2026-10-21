# Revisione bottom-up

Il notebook [03_bottom_up_experiments.ipynb](../../notebook/03_bottom_up_experiments.ipynb)
implementa il confronto OFAT su 12 baseline: riferimento e nove varianti per
numero di esempi (0/2/6 contro 4), coppia di soglie (0.64/0.76, 0.72/0.84,
0.76/0.88 contro 0.68/0.80) e temperatura degli evaluator (0.0/0.4/0.6 contro 0.2).
Sono 120 run per ripetizione. `REPETITIONS` permette ripetizioni indipendenti;
`SUITE_ID` identifica una suite nuova o da riprendere.

Il runner è incorporato nelle celle di supporto del notebook 03. Congela gli
input top-down e registra configurazioni, esempi esatti, hash degli input,
dei moduli della pipeline e delle celle di supporto in
`output/bottom_up_experiments/<suite>/manifest.json`. Ogni configurazione,
ripetizione, dataset e tentativo ha una directory separata, con stato della run,
risultato completo e checkpoint delle iterazioni. Non vengono cancellati output
precedenti. Le run completate, anche non convergenti, vengono saltate alla ripresa;
quelle fallite o interrotte ripartono dalla baseline in un nuovo tentativo.
Anche le funzioni per esportare le iterazioni per configurazione e dataset
sono incorporate nel notebook: non sono più necessari `experiments.py` e
`iteration_exports.py` in `src/bottom_up/`. Il codice delle celle di supporto
viene conservato per provenienza in `notebook_helpers_snapshot.py` nella suite.
Le suite prodotte con i moduli precedenti restano leggibili dal notebook 04;
il nuovo runner richiede una nuova suite per registrarne la diversa provenienza.

Le varianti sono parametri espliciti di `run_global_goal_cycle`:
`evaluator_few_shot_examples`, `evaluator_temperature`, `parent_threshold` e
`duplicate_threshold`. I default della pipeline esistente restano invariati.
Esempi e temperatura variano solo per i tre evaluator globali: il notebook
mantiene generazione zero-shot, reflection con quattro esempi e limiti originali.

La sezione finale carica le ground truth solo per la valutazione esterna,
riutilizzando la metrica soft del notebook di confronto (BERT, matching Hungarian,
soglia fissa 0.65). Esporta CSV di metriche HLG/LLG, delta dalla baseline,
confronti appaiati con il riferimento, macro-medie, variabilità tra ripetizioni,
validità delle risposte, quorum e operazioni, con numerosità e stati delle run.
Gli hash e uno snapshot delle ground truth sono salvati nel manifest di valutazione.
I risultati non convergenti rimangono valutabili e identificati; errori e run
mancanti non vengono trasformati in gerarchie con punteggio zero.

La policy corrente è `cosine_global_documentary_review_v58_three_limits`.
La descrizione del metodo e la tabella decisionale sono nel
[README principale](../../README.md).

Il modulo `goal_reconstructor.py` ricostruisce HLG′ dai soli LLG.
`goal_similarity.py` calcola embedding e cosine similarity, assegnando le
relazioni numeriche e i candidati duplicati dello stesso attore.
Calcola inoltre ogni HLG′ contro tutti gli HLG originali dello stesso attore,
incluso il proprio, e registra `original_hlg_comparisons` per branch. Il filtro
precede il calcolo e normalizza maiuscole e spazi del nome. `build_cosine_payload`
prepara l'oggetto salvato in `<progetto>_cosine_NNN.json` e passato identico ai tre critic.
Gli originali dello stesso attore restano contesto per le decisioni. Il prompt
ammette solo KEEP_ORIGINAL_HIGH_LEVEL_GOAL, REWRITE_ORIGINAL_HIGH_LEVEL_GOAL,
REMOVE_ORIGINAL_HIGH_LEVEL_GOAL e DUPLICATE_REMOVAL. KEEP può richiedere
regenerate_llgs=true con regeneration_feedback; non costituisce una quinta azione.
Non vengono generati nuovi HLG. EVALUATION_INCONCLUSIVE è uno stato interno
dell'orchestratore per voti mancanti, invalidi o senza quorum, non una scelta LLM.
Separatamente, HLG′ simili ad altri HLG′ dello stesso attore con cosine ≥ 0.80
ricevono `duplicate_label: "DUPLICATE"`, con identificativi in `duplicate_candidates`.
Senza candidati l'etichetta è `null`; `relationship` resta il confronto con il padre.
L'etichetta segnala candidati e non autorizza da sola una rimozione.
`global_goal_evaluator.py` invia un unico prompt globale ai tre evaluator:
le risposte riguardano la documentazione e le azioni, non la similarità.
Le richieste usano JSON Schema su `/v1/chat/completions` di LM Studio, con
campi specifici per azione e identificativi ammessi. Il validatore numerico
determina anche l'ammissibilità delle azioni nello schema. I retry richiedono
soltanto i branch irrisolti e conservano il contesto globale e i voti validi.
La cronologia registra in `generation` il motivo di arresto e l'uso dei token;
risposte interrotte o vuote restano invalide, senza fallback a testo libero.
`goal_cycle_orchestrator.py` applica le correzioni e rivaluta l'intero snapshot.

`evaluate_all_branches` restituisce un dizionario di risultati per branch.
`run_global_goal_cycle` mantiene l'interfaccia con baseline HLG/LLG e aggiunge
il parametro facoltativo `actors` per trasmettere tutti gli attori noti.
Il risultato del ciclo registra la configurazione; `<progetto>_iteration_NNN.json` conserva
`similarity_analysis`, input, output, voti, stato delle riparazioni e `stage_logs`.
Ogni iterazione produce solo questo file e `<progetto>_cosine_NNN.json`, senza file per
modello o branch. Il notebook passa `verbose=True` per mostrare confronti,
decisioni per modello e branch, errori, quorum e riparazioni. Le chiamate dirette
al ciclo mantengono `verbose=False` come default, con i log delle fasi in `stage_logs`.
Il notebook gestisce direttamente una cartella per progetto sotto
`output/bottom_up_iterations/`, con il nome leggibile del progetto. All'avvio di
`run_all_datasets()` pulisce i risultati precedenti in questa radice, dopo aver
validato il percorso. Conserva tutte le iterazioni dell'esecuzione corrente.

Non vengono eseguiti controlli diagnostici LLG separati, fusioni dei duplicati
o ricerche globali di HLG mancanti. Le correzioni richiedono generazione e
reflection approvate prima di modificare la gerarchia.
I tipi storici condivisi in `data_model.py` restano
disponibili per leggere i risultati esistenti, senza attivarne la logica.

La soglia HLG–HLG′ predefinita è 0.68, ricavata preliminarmente dalle otto coppie
annotate di Project 25–26. `src/bottom_up/similarity_calibration.py` seleziona
la soglia offline; le annotazioni non vengono lette durante il ciclo. La soglia
dei duplicati resta 0.80. La configurazione distingue calibrazione preliminare,
eventuale override e assenza di validazione indipendente.

Il prompt usa quattro esempi sintetici e un JSON cosine compatto: i confronti
indicano il padre mediante original_branch_index, contando da zero nella lista
branches, e omettono soltanto lo stato VALID e i metadati ripetuti. I punteggi
restano invariati e similarity_analysis conserva i confronti completi. Gli errori
di capienza non vengono ritentati identicamente; uno snapshot invariato bloccato
da questi errori termina senza convergenza.

La bottom-up non esegue audit diretti degli LLG o diagnosi della fedeltà
della ricostruzione. La valutazione qualitativa resta ai due critic originali.
Le rigenerazioni richieste dalle regole richiamano generazione e reflection
originali, senza introdurre un nuovo stadio di valutazione dei figli.

I soli limiti operativi sono 25 iterazioni globali, 3 cicli di riscrittura per
discendenza HLG e 3 tentativi di generazione/reflection per fase a soglia 8.5/10.
Le rigenerazioni LLG e i rifiuti vengono contati senza un limite dedicato;
restano soggetti al massimo globale di iterazioni.
