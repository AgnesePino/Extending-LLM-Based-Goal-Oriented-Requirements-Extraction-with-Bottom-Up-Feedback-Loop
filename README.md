# Estrazione e revisione di obiettivi funzionali con LLM

Il progetto estrae attori, obiettivi di alto livello (**HLG**) e obiettivi
di basso livello (**LLG**) dalla descrizione di un sistema. L'estensione
bottom-up verifica e corregge una gerarchia già prodotta dalla pipeline top-down.

La baseline top-down e il risultato bottom-up sono salvati separatamente.
Il bottom-up non corregge automaticamente i riferimenti della baseline:
ogni LLG deve contenere esattamente uno degli HLG ufficiali come parent.

## Percorso di esecuzione

| Notebook | Funzione |
| --- | --- |
| [Top-down](notebook/01_pipeline_execution_top_down_only.ipynb) | Estrae attori, HLG e LLG e salva la baseline |
| [Bottom-up](notebook/01_pipeline_execution_bottom_up_only.ipynb) | Ricostruisce HLG′, revisiona globalmente i padri, verifica i figli dove necessario e cerca risultati mancanti |
| [Valutazione](notebook/02_experimental_evaluation_top_down_vs_bottom_up.ipynb) | Confronta baseline e risultati riusciti con la ground truth |

Il notebook di valutazione include tutte le baseline con ground truth, anche
quando il risultato bottom-up è assente o fallito. Le metriche dei metodi
revisionati richiedono collezioni finali valide; i delta usano coppie disponibili.
`dataset_availability.csv` elenca tutti i dataset registrati e gli output mancanti.
La selezione sperimentale è indipendente dai sei input scelti per l’esecuzione.

## Revisione bottom-up

La logica corrente, `reconstruction_first_critic1_v31`, affronta la sovrapproduzione
e la frammentazione degli HLG insieme alle incoerenze delle decomposizioni.
Un HLG rappresenta un risultato funzionale autonomo per un attore; i suoi LLG
descrivono le interazioni o gli effetti che lo realizzano. Un **branch** comprende
il padre e i suoi figli. Non viene imposto un numero finale di obiettivi.

La revisione dispone della gerarchia completa già generata dal top-down e può
confrontare tra loro i padri e le decomposizioni effettive. La fonte delle
decisioni è la descrizione del progetto: la ground truth resta esterna al ciclo
ed è utilizzata nella valutazione sperimentale.

Ogni iterazione ricostruisce e rivaluta **tutti i branch**, comprese le conferme
precedenti. I giudizi si riferiscono allo stesso stato, prima delle modifiche.

| Stadio | Compito | Decisioni |
| --- | --- | --- |
| Ricostruzione HLG′ | Qwen sintetizza l’intenzione di ciascuna decomposizione ricevendo soltanto nomi e descrizioni dei figli | Obiettivo diagnostico; per branch vuoti la generazione viene evitata |
| Critic 1 globale | Dopo la ricostruzione, ogni modello riceve fonte, lista completa degli HLG e tutte le ricostruzioni in un’unica richiesta | Prima diagnosi HLG–HLG′ rispetto alla fonte; poi KEEP, REWRITE o REMOVE per ogni padre |
| Critic 2 | Verifica direttamente i figli dei padri mantenuti quando la diagnosi o i limiti operativi lo richiedono | KEEP oppure rigenerazione dell’intera decomposizione |
| Discovery della completezza | Confronta fonte, HLG correnti e attori noti quando tutti i branch sono confermati o al limite LLG dopo rivalutazione | Nessuna lacuna oppure richieste di risultati autonomi mancanti; i branch al limite restano irrisolti |

### Qualità dei padri e ruolo di HLG′

Critic 1 inizia dal confronto HLG–HLG′ rispetto alla descrizione, distinguendo
un possibile difetto della decomposizione da un difetto del padre. Valuta poi
autonomia, granularità, duplicati e coperture sulla lista completa. Un padre
documentato non viene ristretto per adattarlo a figli incompleti: quel caso
richiede il controllo LLG. Per KEEP deve spiegare quale risultato indipendente
rappresenti il goal e cosa lo distingua dagli altri: essere documentato o
coerente con i figli non è sufficiente.

| Motivo di rimozione | Criterio |
| --- | --- |
| `FULLY_REDUNDANT` | Un altro HLG dello stesso attore copre interamente il risultato; sono richiesti ID e nome del superstite |
| `OVER_SPECIFIC` | Il goal è una funzione interna a un risultato complessivo; può essere rimosso anche senza un padre coprente |
| `UNSUPPORTED` | L’intenzione è interamente priva di supporto documentale |
| `NON_FUNCTIONAL` | Il goal esprime un puro vincolo senza una capacità funzionale autonoma |

Un REWRITE può correggere un padre o consolidare frammenti complementari dello
stesso attore. Il gruppo viene sostituito con un unico padre e una decomposizione
completa soltanto dopo l’accettazione dei candidati. Anche i branch già
confermati possono essere rivalutati e coinvolti nei nuovi interventi.

La diagnosi HLG′ viene formulata nella prima parte dello stesso Critic 1,
separatamente dall’azione strutturale. Un padre duplicato o troppo operativo
può avere figli coerenti: una relazione `EQUIVALENT` non impedisce quindi la
rimozione. Viceversa, un HLG′ che corrisponde a un altro padre può indicare
figli errati senza invalidare un originale autonomo e documentato.

Per un padre mantenuto servono almeno due diagnosi valide. Se almeno una è
non equivalente, si richiede il controllo diretto LLG; se tutte sono equivalenti
e i figli sono presenti, è possibile la conferma ordinaria. Diagnosi insufficienti
restano inconclusive. HLG′ è una sintesi fallibile: il secondo critic verifica
le differenze sugli LLG effettivi e sulla fonte.

Le due parti sono istruzioni dello stesso messaggio, non chiamate separate.
Resta obbligatoria la verifica diretta dei figli accettati dopo una riparazione,
anche con equivalenza. La discovery finale continua a cercare outcome mancanti.
Le run storiche mantengono la politica registrata nei loro metadati.

### Voto, riparazioni e priorità

Ogni compito valutativo usa Ministral, Granite e Gemma, con quattro esempi
few-shot, senza memoria conversazionale o voti dei peer. L’azione richiede
almeno due voti validi concordi. Risposte mancanti o invalide non diventano KEEP
e non riducono il quorum. Un difetto locale invalida il voto sul branch
interessato; JSON non interpretabili o ID duplicati o estranei invalidano
la risposta globale.

Una lista JSON completa restituita dal Critic 1 viene normalizzata nel campo
`decisions`, mantenendo risposta grezza e controlli sugli identificativi.
Log e trace distinguono con `evaluation_status` i controlli rinviati per modifiche
strutturali, i mancati quorum con voti invalidi, i disaccordi e le diagnosi
insufficienti. Un semplice rinvio non viene segnalato come valutazione fallita.

Due votanti possono richiedere REMOVE indicando coperture diverse; il riferimento
scelto deve superare i controlli applicativi. Per consolidare è invece necessario
l’accordo sullo stesso gruppo. I punteggi non vengono mediati: negli HLG sono
derivati dal codice (5 per KEEP equivalente, 4 per KEEP non equivalente, 2 per
REWRITE o REMOVE); negli LLG, 5 indica mantenimento e 0–3 rigenerazione;
nella completezza, 4–5 indica assenza di lacune e 0–3 richieste di integrazione.

Le rimozioni ammissibili precedono le riscritture. `FULLY_REDUNDANT` e
`OVER_SPECIFIC` non hanno un limite numerico per iterazione, mentre le altre
rimozioni, le riscritture e le rigenerazioni LLG ordinarie hanno ciascuna un
limite di due. Le proposte strutturali aggregate differiscono il normale
controllo LLG, dando priorità alla revisione dei padri.

I controlli proteggono da cancellazioni reciproche e perdita del superstite.
Per rimozioni ridondanti, i figli delle sorgenti e del target entrano nel
contesto della riparazione; se il candidato è respinto, lo stato precedente
viene conservato. `OVER_SPECIFIC` elimina invece il branch senza sostituzione
immediata: la discovery valuta successivamente se manchi un risultato più ampio.

Le riparazioni usano Qwen e un controllo separato mediante Ministral, con soglia
8,5 su 10 e fino a tre tentativi guidati dal feedback. Ricevono fonte,
motivazione, ambito dell’intervento e capacità da preservare.

### Completezza e condizioni di arresto

La discovery è obbligatoria e cerca risultati autonomi mancanti; non certifica
l’assenza di duplicazioni o la granularità dei padri. Non riceve figli,
ricostruzioni o giudizi precedenti. Un passo mancante in un risultato già
coperto riguarda la decomposizione, non richiede automaticamente un nuovo HLG.

La maggioranza favorevole autorizza la raccolta delle richieste dei modelli
favorevoli, raggruppate per attore: non è richiesto un accordo su ogni singola
proposta. Si possono generare al massimo tre nuovi HLG per iterazione, entro
gli attori noti, con relativi figli. Le integrazioni si applicano in passaggi
senza rimozioni o sostituzioni HLG e vengono poi revisionate. Se tutti i padri
sono rimossi, la discovery può operare sulla lista vuota.

Dopo tre riscritture nella stessa discendenza, un ulteriore REWRITE diventa
KEEP operativo con verifica LLG obbligatoria. Ogni branch dispone di due
rigenerazioni LLG e l’ultima decomposizione viene comunque rivalutata. Dopo
due cicli di riparazione semanticamente respinti viene registrata una conferma
forzata: ulteriori riparazioni sono bloccate, ma una rimozione motivata resta
possibile. Il notebook limita il ciclo a venticinque iterazioni.

La convergenza operativa richiede branch confermati secondo queste politiche,
nessuna lacuna e nessuna modifica strutturale da rivalutare. Deve essere letta
insieme a warning e conferme forzate: non equivale automaticamente all’approvazione
semantica di tutti i goal. L’efficacia va misurata separatamente su precisione,
copertura e capacità conservate o perse.

Il [README bottom-up completo](src/bottom_up/README_bottom_up.md) contiene
prompt inglesi, esempi e dettagli implementativi.

## Avvio

1. Installare le dipendenze di [requirements.txt](requirements.txt) nell'ambiente Python usato dai notebook.
2. Avviare il server locale LM Studio e rendere disponibili i modelli configurati in [llm_clients.py](src/llm_clients.py).
3. Eseguire i notebook nell'ordine indicato, usando gli output top-down già disponibili se la baseline è stata generata.
4. Nel notebook bottom-up impostare `DATASETS_TO_RERUN` e `BASELINE_FILE_OVERRIDES` per selezionare gli input.

L'endpoint locale predefinito è `http://127.0.0.1:1234/v1`. Endpoint e modelli
sono configurabili tramite `LM_STUDIO_BASE_URL`, `QWEN_GENERATOR_MODEL`,
`MINISTRAL_CRITIC_MODEL`, `GRANITE_CRITIC_MODEL` e `GEMMA_CRITIC_MODEL`.
Riavviare il kernel dopo modifiche ai moduli o agli schemi dei dati.

## Output e controlli

| Percorso | Contenuto |
| --- | --- |
| `output/top_down/` | Baseline originali |
| `output/top_down_bottom_up/` | Risultati bottom-up o artefatti di errore |
| `output/bottom_up_iterations/` | Trace delle iterazioni e voti dei critic |
| `output/evaluation/three_way_pipeline_comparison/` | Metriche, confronti e grafici |

Per i risultati revisionati leggere `finalHighLevelGoals`, `finalLowLevelGoals`
e `globalGoalCycle`. Le collezioni strutturate iniziali restano come provenienza.
Un errore di esecuzione è distinto da un ciclo terminato con `converged=False`.

Il modulo `hierarchy_audit` non viene eseguito dalla pipeline corrente. Eventuali
campi e report di audit presenti negli artefatti storici appartengono alle
versioni che li hanno prodotti. Le collezioni finali e i normali trace del ciclo
restano disponibili per analizzare le modifiche.

Dentro la cartella di ogni dataset, `ministral/`, `granite/` e `gemma/`
contengono le decisioni individuali per iterazione e branch. `critic1_global.json`
conserva la risposta completa del modello; i voti per branch distinguono `hlg`,
`reconstruction` e `llg`. Il trace registra `critic1_global_votes`,
`hlg_structure_summary` e `structural_action_outcomes`, con proposte, modifiche
applicate o differite e relative motivazioni. I nuovi campi hanno valori
predefiniti per leggere anche artefatti storici senza reinterpretarne i voti.
Le conferme precedenti vengono rivalutate a ogni iterazione.

Il rerun sovrascrive risultati e trace dei dataset selezionati: archiviare gli
artefatti necessari ai confronti sperimentali prima di rieseguire.

I test locali si eseguono con:

```powershell
python -m unittest discover -s tests -v
```

I test simulano le chiamate ai modelli e verificano il comportamento del codice;
non misurano la qualità semantica delle risposte reali.

## Documentazione e risultati

| Documento | Contenuto |
| --- | --- |
| [README bottom-up](src/bottom_up/README_bottom_up.md) | Comportamento corrente della revisione |

I risultati storici descrivono le versioni che li hanno prodotti. Per misurare
gli effetti della logica corrente occorrono nuove esecuzioni sugli stessi input.

Il progetto deriva dalla [pipeline originale](https://github.com/dadoluca/LLM_MultiAgentChain_GoalOrientedAPIAlignement)
e dalle sue successive estensioni. Licenza: [GNU GPL v3](LICENSE).
