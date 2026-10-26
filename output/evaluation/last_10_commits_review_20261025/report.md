# Confronto degli ultimi 10 commit

Analisi dei file versionati con `git show`, senza cambiare checkout e senza eseguire nuovamente gli LLM. Le modifiche locali e i risultati non tracciati sono esclusi dalla graduatoria dei commit.

## Risultato principale

Nella serie storica con lo stesso protocollo, la policy `reconstruction_first_critic1_v31`, valutata in `ef242661` e conservata invariata in `eeb36950`, ha i risultati migliori: F1 macro HLG 0,546094 e LLG 0,588987 su 12 dataset. Baseline top-down: HLG 0,515913 e LLG 0,513146. Incrementi: +3,02 e +7,58 punti percentuali.

## Tutti i commit (dal pi? recente)

| Commit | Policy dei risultati | F1 macro HLG | F1 macro LLG | Evidenza |
|---|---|---:|---:|---|
| `ae11904f` | vedi nota | ? | ? | inherits v63 evaluation committed in bf39b899; distinct metric protocol |
| `bf39b899` | vedi nota | ? | ? | inherits v63 evaluation committed in bf39b899; distinct metric protocol |
| `2a60b0df` | vedi nota | ? | ? | no committed aggregate evaluation |
| `ace5441e` | cosine_global_documentary_review_v58_three_limits | 0.519418 | 0.540155 | historical comparable cohort |
| `b612318d` | cosine_global_documentary_review_v58_three_limits | 0.519418 | 0.540155 | historical comparable cohort |
| `4275af9d` | cosine_global_documentary_review_v58_three_limits | 0.519418 | 0.540155 | historical comparable cohort |
| `a3be1968` | cosine_global_documentary_review_v55_hlg_scope | 0.521017 | 0.529208 | historical comparable cohort |
| `09c87a32` | cosine_global_documentary_review_v55_hlg_scope | 0.521017 | 0.529208 | historical comparable cohort |
| `eeb36950` | reconstruction_first_critic1_v31 | 0.546094 | 0.588987 | historical comparable cohort |
| `ef242661` | reconstruction_first_critic1_v31 | 0.546094 | 0.588987 | historical comparable cohort |

I valori ripetuti non costituiscono repliche indipendenti. Per esempio, `a3be1968` cambia il codice ma mantiene la valutazione della policy v55 presente in `09c87a32`; `b612318d` e `ace5441e` mantengono le metriche di `4275af9d`. In `09c87a32` il file Python contiene gi? riferimenti a v58, ma i metadati della valutazione dichiarano v55: per attribuire i risultati si sono usati questi ultimi.

## Cosa cambia tra le versioni

- `ef242661` / v31: diagnosi semantica della ricostruzione tramite LLM, revisione globale dei padri, audit e rigenerazione LLG quando necessari, analisi finale delle capacit?/HLG mancanti. I dati valutati rappresentano il percorso top-down + bottom-up + completeness.
- `09c87a32`: introduce `goal_similarity.py` e la revisione documentale con etichette numeriche fisse; la valutazione salvata ? v55. Rispetto a v31, F1 macro cala di 2,51 punti HLG e 5,98 punti LLG.
- `a3be1968`: rimuove molto codice precedente dall'evaluator; le metriche salvate restano quelle di v55.
- `4275af9d` / v58: aggiunge tracciamento/configurazione della temperatura e cambia logica/prompt. Rispetto alla valutazione v55, HLG -0,16 punti e LLG +1,09 punti. Il ciclo nuovo non chiama pi? la funzione di analisi degli HLG mancanti presente in v31; l'evaluator non esegue lo stesso audit LLG di v31. Il confronto cambia quindi pi? fattori contemporaneamente e non isola l'effetto del prompt.
- `2a60b0df` / v63: modifica anche il protocollo di valutazione e introduce utility per valutare descrizioni e aggregare metriche pesate; elimina i precedenti report storici. Non contiene una nuova tabella aggregata confrontabile.
- `bf39b899` / codice v69: salva un esperimento completo prodotto da v63 e una suite v69 incompleta (32/72 run). Il report completo non misura il nuovo codice v69.
- `ae11904f` / codice v105: modifica prompt, ciclo, ricostruzione, similarit? e storage, ma i CSV versionati sono gli stessi di `bf39b899`; non dimostrano un miglioramento di v105. Anche il riepilogo versionato della suite v69 ? ancora incompleto.

## Esperimento completo v63 salvato in bf39b899

12 dataset, una ripetizione per configurazione, soglie parent e duplicate 0,68.

| Configurazione | Esempi | Temperatura | F1 pesata HLG | F1 pesata LLG |
|---|---:|---:|---:|---:|
| E1_0_examples | 0 | 0,2 | 0,564678 | 0,539970 |
| E1_1_examples | 1 | 0,2 | 0,576343 | **0,560264** |
| reference | 4 | 0,2 | 0,577933 | 0,551363 |
| E2_temperature_0.0 | 4 | 0,0 | 0,556435 | 0,542137 |
| E2_temperature_0.4 | 4 | 0,4 | 0,572441 | 0,550514 |
| E2_temperature_0.6 | 4 | 0,6 | **0,579109** | 0,553510 |

Non c'? un vincitore unico per entrambi i livelli: temperatura 0,6 massimizza HLG; un esempio a temperatura 0,2 massimizza LLG. Non sono stime di superiorit? statistica, essendoci una sola ripetizione. La configurazione `reference` indica quella di riferimento sperimentale, non la ground truth.

## Perch? non si pu? fare una graduatoria unica

Il manifest storico dichiara: `bert-base-uncased`, testo del nome del goal (descrizione come fallback), testo grezzo senza stemming/lemmatizzazione, assegnamento ungherese con soli match cosine >= 0,65 e F1 macro sui dataset.

Il manifest del report v63 dichiara: stesso modello nominale, descrizioni, tokenizzazione/stopword/stemming/lemmatizzazione NLTK, tutti i match assegnati senza soglia e aggregazione principale pesata sul numero di goal di riferimento. Sono diverse anche le baseline salvate (verificate mediante gli hash dei manifest). Anche usare le macro del nuovo esperimento non rende omogenei i protocolli.

Conclusione operativa: v31 ? il miglior riferimento storico nella serie omogenea; per scegliere tra v31, v63 e v105 occorre rivalutare gli output salvati sul medesimo insieme di baseline/ground truth, con lo stesso scorer e la stessa aggregazione. I dati attuali non attribuiscono causalmente il calo a una singola modifica.

## Provenienza

Le copie CSV nella stessa cartella provengono dai rispettivi oggetti Git. Fonti originali: `output/evaluation/three_way_pipeline_comparison/{macro_comparison.csv,evaluation_summary.json}` nei commit storici; `output/evaluation/configuration_comparison/20261023_090732_305510/{weighted_metrics.csv,macro_metrics.csv,configurations.csv,evaluation_manifest.json}` e i `results_summary.json` delle suite nei commit recenti.
