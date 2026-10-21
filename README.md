# Estrazione e revisione di obiettivi funzionali con LLM

## Introduzione

L'estensione implementata in questo progetto consiste in una pipeline bottom-up basata su un'architettura multi-agente, sviluppata per revisionare la gerarchia di obiettivi funzionali prodotta dalla pipeline descritta nell'articolo *Evaluating LLM-Based Goal Extraction in Requirements Engineering: Prompting Strategies and Their Limitations* di Arnaudo et al. Il processo interviene dopo l'estrazione iniziale e riutilizza i componenti di generazione e reflection per applicare le correzioni, senza sovrascrivere il file della baseline top-down.

La pipeline di riferimento parte dalla descrizione del progetto, identifica gli attori, genera gli High-Level Goals (HLG) e li decompone in Low-Level Goals (LLG). Gli HLG rappresentano i risultati funzionali perseguiti dagli attori, mentre gli LLG descrivono le capacità e le attività necessarie a realizzarli. Durante la generazione, i critic della pipeline originale valutano gli obiettivi e forniscono feedback ai successivi tentativi. La verifica diretta della qualità degli LLG rimane affidata a questi componenti e non costituisce un'ulteriore attività dell'ensemble bottom-up.

La revisione bottom-up considera ciascun HLG originale e i relativi LLG come un branch. A partire esclusivamente dai nomi e dalle descrizioni dei figli, Qwen3-VL-8B ricostruisce un possibile obiettivo sovraordinato, denominato HLG′, che rappresenta l'intenzione funzionale espressa dalla decomposizione. Il ricostruttore non riceve il padre originale, la documentazione, il registro degli attori o gli altri branch.

Una volta completate le ricostruzioni, il codice calcola la cosine similarity mediante gli embedding di all-mpnet-base-v2. Ogni HLG′ viene confrontato con il proprio padre, con tutti gli HLG originali dello stesso attore e con le altre ricostruzioni dello stesso attore. I confronti producono la relazione EQUIVALENT o NOT_EQUIVALENT con il padre, le corrispondenze con gli altri originali e le coppie candidate alla duplicazione. Le etichette numeriche sono stabilite dal codice e non vengono rivalutate dagli LLM.

La documentazione costituisce il riferimento per decidere quale intervento sia giustificato a partire da questi esiti. Per ciascuno snapshot, Ministral-3-14B-Reasoning, Granite-3.2-8B e Gemma-4-12B-QAT ricevono lo stesso prompt globale, contenente la fonte, gli attori, tutti gli HLG e HLG′ e i confronti numerici. Ogni modello propone una decisione per branch, indipendentemente dagli altri evaluator. La ground truth rimane esterna alle decisioni ed è utilizzata per il confronto sperimentale.

Gli evaluator possono proporre soltanto quattro operazioni: mantenimento, riscrittura, rimozione e rimozione per duplicazione. Il mantenimento può essere accompagnato dalla rigenerazione degli LLG quando il padre è valido ma HLG′ non è supportato dalla fonte. Non sono previste aggiunte di nuovi HLG. La bottom-up non introduce una diagnosi della fedeltà della ricostruzione o una ricerca globale di obiettivi mancanti.

L'implementazione corrente corrisponde alla policy **cosine_global_documentary_review_v58_three_limits**. La revisione documentale utilizza un unico tipo di prompt, inviato separatamente ai tre evaluator; non è articolata in tre prompt diagnostici successivi.

## Organizzazione del processo bottom-up

Ogni iterazione analizza uno snapshot immutabile della gerarchia. L'orchestratore raccoglie le risposte JSON dei tre evaluator, ne verifica la struttura, i campi richiesti, i riferimenti ai branch e la compatibilità con le classificazioni numeriche. Le decisioni vengono aggregate mediante majority voting, richiedendo almeno due voti validi concordi su tre per ciascuna operazione. Le risposte invalide non riducono il quorum e non vengono considerate conferme.

Dopo l'aggregazione, il mantenimento conserva il branch e la rimozione elimina padre e figli. Le riscritture e le rigenerazioni richiamano Qwen3-VL-8B per la generazione e Ministral-3-14B-Reasoning per la reflection. Una modifica generativa viene applicata soltanto quando i candidati e i controlli previsti sono accettati; in caso contrario rimane la gerarchia precedente. Le operazioni incompatibili vengono rinviate secondo l'ordine descritto nelle sezioni successive.

Un branch senza maggioranza rimane invariato e non viene confermato. Dopo una modifica, tutti i branch sopravvissuti vengono nuovamente ricostruiti e valutati. Il processo converge soltanto dopo un passaggio completo senza modifiche e con tutti i branch confermati; altrimenti prosegue entro i limiti operativi oppure termina senza convergenza in presenza delle condizioni bloccanti previste.

La tabella seguente riassume gli input, i modelli coinvolti e i risultati delle principali fasi. Gli identificativi riportati sono quelli predefiniti in configurazione; eventuali sostituzioni vengono registrate negli output dell'esecuzione.

| Fase | Informazioni fornite | Modello o componente | Risultato |
| --- | --- | --- | --- |
| Ricostruzione HLG′ | Nomi e descrizioni dei soli LLG del branch | Qwen3-VL-8B — `qwen/qwen3-vl-8b` | Un HLG′ per ogni branch non vuoto |
| Confronto numerico | Nome e descrizione degli HLG originali, testo degli HLG′ e attribuzione agli attori | All-MPNet-Base-v2 — `sentence-transformers/all-mpnet-base-v2`, con calcolo cosine nel codice | Punteggi, relazione con il padre, corrispondenze con gli altri originali e candidati duplicati |
| Revisione documentale globale | Documentazione, attori, tutti gli HLG e HLG′, punteggi ed etichette numeriche | Ministral-3-14B-Reasoning — `mistralai/ministral-3-14b-reasoning`; Granite-3.2-8B — `ibm/granite-3.2-8b`; Gemma-4-12B-QAT — `google/gemma-4-12b-qat` | Una decisione documentale per branch e per modello, in JSON strutturato |
| Validazione e aggregazione | Risposte dei tre evaluator e vincoli dell'operazione | Orchestratore e validatori nel codice, senza un ulteriore LLM | Operazione con almeno due voti validi concordi oppure esito inconclusivo |
| Generazione delle correzioni | Fonte, attore, perimetro del padre e feedback dell'intervento | Qwen3-VL-8B — `qwen/qwen3-vl-8b` | Candidato HLG e/o decomposizione LLG, secondo l'operazione |
| Reflection delle correzioni | Candidato generato e contesto della riparazione | Ministral-3-14B-Reasoning — `mistralai/ministral-3-14b-reasoning`, tramite i critic originali di HLG e LLG | Candidato accettato con soglia 8.5/10 oppure feedback, entro tre tentativi per fase |
| Applicazione e nuova iterazione | Decisioni aggregate e correzioni accettate | Orchestratore; nuova ricostruzione con `qwen/qwen3-vl-8b` e nuova revisione con i tre evaluator sopra indicati | Gerarchia rivalutata fino alla convergenza o all'arresto senza convergenza |

## Ricostruzione HLG′ e significato del confronto inverso

La ricostruzione viene eseguita separatamente per ciascun branch mediante Qwen, che riceve esclusivamente i nomi e le descrizioni degli LLG associati. Il modello non vede l'HLG originale, la documentazione del progetto, il registro degli attori o gli altri branch. L'intenzione ricostruita deve quindi derivare dalle informazioni presenti nei figli, senza essere suggerita dal padre.

HLG′ rappresenta la sintesi della decomposizione effettiva e costituisce il punto di partenza del confronto con il padre. La bottom-up utilizza questa rappresentazione per decidere gli interventi sugli obiettivi sulla base delle etichette numeriche e della documentazione.

Per un branch privo di LLG non viene chiamato il ricostruttore. Il codice registra l'assenza della decomposizione, senza calcolare una similarità rispetto a un testo sostitutivo. Il punteggio rimane assente e la relazione con il padre è NOT_EQUIVALENT.

## Confronto numerico mediante cosine similarity

Il confronto viene effettuato utilizzando sentence-transformers/all-mpnet-base-v2. Per ciascun padre vengono rappresentati congiuntamente nome e descrizione, mentre per HLG′ viene utilizzato il testo ricostruito. Gli embedding sono normalizzati; non vengono applicati stemming o rimozione delle stopword.

Il primo confronto riguarda ogni HLG′ e il proprio padre. Una cosine almeno pari a **0.68** determina l'etichetta EQUIVALENT; un valore inferiore determina NOT_EQUIVALENT. Queste etichette esprimono l'esito della soglia numerica e non costituiscono una prova di equivalenza funzionale completa o di supporto documentale.

Il secondo confronto riguarda ogni HLG′ e tutti gli HLG originali dello stesso attore, incluso il proprio padre. Le corrispondenze con gli altri padri sono individuate mediante la soglia 0.68 e registrate in matching_other_original_hlg_ids. Questi confronti costituiscono contesto per la revisione della gerarchia e non attivano la creazione di nuovi obiettivi.

Il terzo confronto riguarda gli HLG′ appartenenti a branch differenti dello stesso attore. Una cosine almeno pari a **0.80** identifica una coppia candidata alla duplicazione. Entrambi i branch ricevono duplicate_label con valore DUPLICATE e gli identificativi dei candidati in duplicate_candidates. In assenza di candidati, l'etichetta è null. Tale segnalazione rimane distinta dalla relazione con il proprio padre e non autorizza automaticamente una rimozione.

Il filtro per attore precede il calcolo dei confronti tra branch e normalizza maiuscole e spazi nel nome dell'attore. Le matrici dei confronti con gli originali sono registrate in original_hlg_comparisons. L'intero oggetto JSON cosine viene salvato e trasmesso identico ai tre evaluator.

Per contenere la lunghezza, il JSON destinato agli evaluator non ripete nome e attore di ciascun padre in ogni cella della matrice. Il campo original_branch_index identifica il padre mediante la sua posizione nella lista branches, contando da zero. Uno stato di confronto omesso indica VALID; errori e punteggi assenti restano espliciti. I valori delle cosine rimangono invariati. La configurazione completa e i confronti con gli identificativi originali restano in similarity_analysis nel file di iterazione. Il prompt presenta quattro esempi sintetici nella forma descrizione, attore, obiettivi e risposta, con scenari originali e senza utilizzare i requisiti della ground truth.

Gli embedding vengono riutilizzati per testi identici e calcolati sulla CPU. Gli input oltre il limite del modello producono un errore esplicito, senza troncamento silenzioso. Anche gli errori delle dipendenze native vengono controllati prima del caricamento mediante un processo separato. Se tutti i confronti risultano indisponibili, il ciclo termina senza convergenza.

## Revisione documentale mediante il prompt globale

Una volta completati i confronti mediante cosine similarity, la pipeline avvia la revisione documentale attraverso un prompt globale, inviato separatamente a Ministral-3-14B-Reasoning, Granite-3.2-8B e Gemma-4-12B-QAT. I modelli ricevono gli stessi input e formulano giudizi indipendenti, senza conoscere le risposte degli altri evaluator.

Il prompt contiene la documentazione originale, gli attori identificati, i nomi e le descrizioni degli HLG originali, le rispettive ricostruzioni HLG′ e i risultati numerici. Ogni evaluator dispone quindi delle informazioni dei singoli branch e del contesto complessivo della gerarchia. Quattro esempi few-shot mostrano il formato delle risposte e il comportamento decisionale atteso. La temperatura predefinita è 0.2.

Il compito degli evaluator non è ricalcolare la cosine similarity o modificarne le classificazioni, ma stabilire quale intervento sia giustificato dalla documentazione. Gli esiti numerici delimitano le operazioni ammesse; i modelli verificano il risultato funzionale espresso dagli obiettivi, la responsabilità dell'attore e l'effettiva ridondanza di eventuali candidati duplicati. Una somiglianza numerica non dimostra da sola che un branch debba essere eliminato.

Il prompt consente soltanto le seguenti quattro operazioni.

| Operazione LLM | Condizione numerica e documentale | Intervento |
| --- | --- | --- |
| KEEP_ORIGINAL_HIGH_LEVEL_GOAL | EQUIVALENT e risultato documentato | Conservare padre e figli |
| KEEP_ORIGINAL_HIGH_LEVEL_GOAL con regenerate_llgs: true | NOT_EQUIVALENT, padre valido e HLG′ non supportato dalla documentazione; oppure branch vuoto con padre valido | Conservare il padre e rigenerare gli LLG mediante generazione e reflection originali |
| REWRITE_ORIGINAL_HIGH_LEVEL_GOAL | NOT_EQUIVALENT, HLG′ documentato e fonte che giustifica una correzione del risultato funzionale del padre | Riscrivere il padre secondo rewriting_focus e generarne la decomposizione |
| REMOVE_ORIGINAL_HIGH_LEVEL_GOAL | EQUIVALENT e risultato interamente non documentato | Eliminare il padre e i suoi figli |
| DUPLICATE_REMOVAL | Coppia candidata dello stesso attore confermata come ridondante dalla fonte | Conservare surviving_branch_id ed eliminare l'altro branch con i figli |

La richiesta di rigenerazione è un parametro di KEEP, non una quinta scelta dell'evaluator. Con NOT_EQUIVALENT, un semplice KEEP senza regenerate_llgs: true e regeneration_feedback non è ammesso. La riscrittura richiede rewriting_focus e confronti con gli originali completi; non può partire dal testo sostitutivo di un branch vuoto.

Per esempio, HLG′ «Gestire il proprio profilo» può correggere il padre «Gestire gli account di tutti i membri» se la fonte limita l'attore alla modifica del proprio profilo. L'assenza di corrispondenze numeriche con gli altri padri non introduce un'operazione di aggiunta.

Per la duplicazione si conserva il padre più completo e correttamente attribuito; a parità vale l'ordine del registro. Il superstite deve appartenere ai candidati diretti ed essere mantenuto dalla valutazione aggregata. Non sono ammesse autorimozioni, rimozioni circolari o deduzioni per sola transitività. I figli eliminati non vengono trasferiti. Se la fonte distingue due risultati autonomi, ciascun branch segue le altre regole e l'etichetta DUPLICATE non comporta una rimozione automatica.

Ogni evaluator restituisce un JSON strutturato con branch_id, decision e una breve rationale documentale per ciascun branch richiesto. Solo quando richiesti dall'operazione si aggiungono rewriting_focus, regenerate_llgs con regeneration_feedback oppure surviving_branch_id. Ogni branch richiesto deve comparire esattamente una volta con una delle quattro azioni. Lo schema richiede il numero esatto di decisioni: non sono ammesse astensioni, voci omesse, risposte vuote o etichette ulteriori. Le risposte mancanti o invalide sono errori da correggere nei tentativi previsti.

EVALUATION_INCONCLUSIVE rimane esclusivamente uno stato del codice, non un'etichetta ammessa nelle risposte degli LLM. In presenza di voti mancanti, invalidi o senza maggioranza, l'orchestratore conserva il branch senza confermarlo. La verifica diretta degli LLG e della fedeltà della ricostruzione resta esterna all'ambito dell'ensemble bottom-up.

## Risposte strutturate e aggregazione dei giudizi

Le risposte vengono vincolate mediante JSON Schema attraverso l'endpoint locale /v1/chat/completions di LM Studio. Ogni risposta contiene una lista decisions, con una voce per ciascun branch richiesto. La voce comprende branch_id, decision e una breve rationale documentale; gli altri campi sono presenti soltanto quando richiesti dall'azione.

Lo schema limita gli identificativi, i campi e le azioni numericamente consentite. Il codice verifica successivamente le decisioni di ciascun branch, mentre il supporto documentale rimane oggetto del giudizio del modello. Una decisione non valida non invalida automaticamente le altre decisioni leggibili e valide della stessa risposta. Un JSON incompleto o non interpretabile non produce invece voti.

Gli errori di capienza del contesto non vengono ripetuti con richieste identiche: il voto rimane invalido e il trace registra CONTEXT_LENGTH_EXCEEDED. Se tali errori impediscono il quorum su uno snapshot invariato, il ciclo termina senza convergenza e conserva la gerarchia disponibile.

Sono previsti al massimo tre tentativi. I voti validi vengono conservati e i tentativi successivi richiedono soltanto le decisioni irrisolte, mantenendo il contesto globale. La cronologia registra il testo ricevuto, gli errori e, quando disponibili, il motivo di arresto e il consumo di token. Una risposta interrotta o vuota viene segnalata esplicitamente; il rifiuto dello schema non determina un passaggio automatico al testo libero.

Il majority voting richiede almeno due modelli distinti concordi sull'azione, sulla richiesta di rigenerazione e sull'eventuale superstite. Non è necessario che le motivazioni abbiano la stessa formulazione. Un voto per KEEP e uno per REWRITE non costituiscono accordo. Due KEEP con richieste diverse di rigenerazione non costituiscono accordo sulla stessa operazione. Errori di formato e voti mancanti non riducono il quorum; senza maggioranza il branch rimane inconclusivo.

## Ritorno alla generazione e applicazione delle modifiche

Dopo l'aggregazione, l'orchestratore applica gli interventi sullo snapshot. Le rimozioni precedono le riscritture. Le rigenerazioni ordinarie degli LLG vengono rinviate in presenza di interventi strutturali sui padri.

Le correzioni riutilizzano i generator e la reflection della top-down. Qwen genera i candidati e Ministral ne valuta la qualità, con soglia 8.5/10 ed entro tre tentativi per fase. Una riscrittura viene applicata soltanto insieme alla decomposizione approvata.

Se la generazione, la reflection o i controlli falliscono, lo stato precedente viene conservato e la riparazione rimane irrisolta.

Ogni LLG deve riferirsi al proprio padre mediante high_level_associated, con nome, descrizione e attore esattamente uguali. Non vengono accettate decomposizioni vuote o associazioni a padri differenti da quello richiesto.

## Orchestratore, iterazione e limiti operativi

L'orchestratore è il componente software che coordina ricostruzione, confronti numerici, richieste agli evaluator, validazione delle risposte, voto e applicazione delle modifiche. Non è un modello linguistico. Ogni passaggio utilizza uno snapshot della gerarchia e gli interventi approvati diventano oggetto della valutazione successiva.

Dopo una modifica, tutti i branch sopravvissuti vengono ricostruiti e valutati nuovamente. L'approvazione di un candidato non conclude pertanto la revisione: la nuova gerarchia deve attraversare un passaggio completo di ricostruzione, confronto numerico e valutazione documentale.

La convergenza richiede un'iterazione senza modifiche e con tutti i branch confermati. L'orchestratore applica soltanto i seguenti limiti operativi:

- **Massimo 25 iterazioni globali**, per limitare i cicli completi di revisione della gerarchia.
- **Massimo 3 riscritture per discendenza HLG**, con il contatore trasferito al padre risultante dalla riscrittura. Ogni avvio del ciclo di riparazione HLG consuma un tentativo di riscrittura della discendenza, anche se il candidato viene respinto.
- **Massimo 3 tentativi di generazione e reflection per fase**, per ottenere una correzione che raggiunga la soglia di accettazione di 8.5/10.

I conteggi delle rigenerazioni LLG e delle riparazioni respinte sono conservati per tracciabilità e non introducono ulteriori soglie di arresto. Il raggiungimento dei limiti non costituisce una conferma automatica. Se il numero massimo di iterazioni viene esaurito senza convergenza, il risultato registra MAX_ITERATIONS_REACHED e converged=false. Se non rimangono interventi eseguibili perché le riscritture consentite sono esaurite, o si verifica una condizione bloccante, il ciclo termina senza convergenza con la motivazione nel trace.

Le decisioni disponibili coprono i principali interventi sulla gerarchia esistente, ma non tutte le possibili incoerenze. Un risultato assente sia dai padri sia dai figli non può emergere da HLG′, poiché il processo non comprende una ricerca globale di completezza. Non sono previste operazioni dedicate di fusione, suddivisione o trasferimento dei figli. Il filtro per attore esclude inoltre i confronti numerici tra attori diversi.

La verifica diretta della qualità degli LLG e la diagnosi della fedeltà della ricostruzione sono esterne all'ambito di questa estensione. La valutazione degli obiettivi generati rimane affidata ai due critic della pipeline originale, riutilizzati anche nella generazione delle riparazioni. La bottom-up non introduce un audit aggiuntivo degli LLG: quando le regole richiedono una nuova decomposizione, ne attiva la generazione e la reflection originali. La convergenza descrive la stabilità raggiunta secondo le regole implementate e non certifica la completezza globale rispetto alla fonte.

## Calibrazione e valutazione sperimentale

La soglia 0.68 deriva da una calibrazione preliminare su otto coppie HLG–HLG′ della prima iterazione di Project 25–26, annotate mediante revisione semantica dell'assistente. Cinque coppie sono state considerate equivalenti e tre diverse. La selezione massimizza F1 della classe equivalente su una griglia con passo 0.01; a parità utilizza balanced accuracy e distanza dal punteggio osservato più vicino.

Su queste coppie la soglia produce cinque veri positivi, due veri negativi e un falso positivo. Le annotazioni non sono indipendenti e non costituiscono una validazione su dati separati. La soglia dei duplicati, pari a 0.80, rimane iniziale e non calibrata.

Il modulo [similarity_calibration.py](src/bottom_up/similarity_calibration.py) permette di ripetere offline la selezione su coppie annotate. Non viene chiamato durante il ciclo e non adatta la soglia al dataset o all'iterazione in corso. I dati numerici della calibrazione preliminare sono conservati anche nei relativi test.

La qualità della revisione può essere valutata confrontando la baseline e il risultato bottom-up con le rispettive ground truth, mediante Precision, Recall e F1-score calcolati separatamente per HLG e LLG. Tale confronto deve essere affiancato dall'analisi delle modifiche effettivamente applicate, distinguendo i benefici attesi dalle prestazioni misurate. I risultati di un'esecuzione si riferiscono alla policy e alla configurazione registrate nei suoi output.

## Notebook e modalità di esecuzione

Il repository separa la produzione della baseline, la revisione e la valutazione sperimentale nei seguenti notebook.

| Notebook | Funzione |
| --- | --- |
| [Top-down](notebook/01_pipeline_execution_top_down_only.ipynb) | Estrazione iniziale di attori, HLG e LLG |
| [Bottom-up](notebook/01_pipeline_execution_bottom_up_only.ipynb) | Revisione delle baseline esistenti |
| [Evaluation](notebook/02_experimental_evaluation_top_down_vs_bottom_up.ipynb) | Confronto delle gerarchie con la ground truth |

Per eseguire il processo occorre selezionare il kernel .venv del progetto e installare le dipendenze di [requirements.txt](requirements.txt), incluso sentence-transformers. Il primo caricamento del modello embedding richiede il download se non è disponibile nella cache. LM Studio deve essere avviato con i modelli configurati in [llm_clients.py](src/llm_clients.py).

Prima dell'avvio del ciclo è necessario eseguire la cella di import/reload del notebook bottom-up, soprattutto dopo una modifica al codice. La baseline viene letta da output/top_down/ e il risultato combinato viene salvato in output/top_down_bottom_up/.

I parametri principali possono essere configurati mediante variabili d'ambiente, da impostare prima dell'importazione oppure prima di ricaricare i moduli.

| Variabile | Valore predefinito |
| --- | --- |
| BOTTOM_UP_HLG_SIMILARITY_THRESHOLD | 0.68 |
| BOTTOM_UP_DUPLICATE_SIMILARITY_THRESHOLD | 0.80 |
| BOTTOM_UP_EMBEDDING_REVISION | main |
| BOTTOM_UP_EVALUATOR_TEMPERATURE | 0.2 |
| BOTTOM_UP_EVALUATOR_VOTE_ATTEMPTS | 3 |

Per la riproducibilità è possibile fissare una revisione del modello embedding. La configurazione registra anche la revisione effettivamente risolta, quando disponibile.

## Organizzazione degli output e tracciabilità

La gestione delle cartelle è definita direttamente nel notebook. Ogni progetto utilizza una cartella con il proprio nome sotto output/bottom_up_iterations/, per esempio **output/bottom_up_iterations/SIA Project 25 26/**. All'avvio di run_all_datasets() vengono cancellati i file e le cartelle delle precedenti esecuzioni in questa radice, dopo la verifica del percorso di destinazione. La pulizia non riguarda la baseline e non avviene tra iterazioni: tutte le iterazioni dell'esecuzione corrente vengono conservate.

Per ciascuna iterazione vengono prodotti due JSON, senza file separati per modello o branch. Il nome del progetto viene utilizzato come prefisso, sostituendo spazi e caratteri non adatti con underscore.

| Esempio di file | Contenuto |
| --- | --- |
| SIA_Project_25_26_cosine_001.json | HLG, HLG′, configurazione embedding, confronti numerici e candidati duplicati; contiene l'oggetto fornito ai tre evaluator |
| SIA_Project_25_26_iteration_001.json | Gerarchia iniziale e finale, ricostruzioni, risposte, tentativi, voti e interventi applicati |

Il JSON cosine viene scritto prima delle richieste ai modelli. Il file di iterazione viene aggiornato come checkpoint dopo ciascun evaluator e completato dopo le correzioni. Questa distinzione permette di ricostruire gli input della valutazione e di verificare se una proposta sia stata applicata, rinviata o respinta.

Il notebook utilizza verbose=True e stampa confronti cosine, modello e tentativo in corso, decisioni e motivazioni per branch, errori, quorum e correzioni applicate. Le risposte e i tentativi rimangono disponibili anche nel JSON. Con verbose=False, il ciclo mostra soltanto i riepiloghi e conserva i log delle fasi in stage_logs. Le chiamate dirette possono impostare output_file_prefix per personalizzare i nomi; omettendolo mantengono i nomi generici delle iterazioni.

## Verifiche dell'implementazione

I test automatici verificano calcoli, soglie, contratti delle risposte, quorum, tentativi di correzione, applicazione delle riparazioni e condizioni di convergenza, simulando le risposte dei modelli. Verificano inoltre la gestione delle cartelle del notebook. Non sostituiscono una valutazione qualitativa degli output reali né dimostrano la validità sperimentale delle soglie.

~~~powershell
python -m pytest -q
~~~

Il progetto deriva dalla pipeline originale di Arnaudo et al. ed è distribuito con licenza GNU GPL v3.
