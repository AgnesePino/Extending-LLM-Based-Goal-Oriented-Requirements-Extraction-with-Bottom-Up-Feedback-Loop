# Analisi delle esecuzioni bottom-up

Suite analizzata: `ofat_fixed_thresholds_20261023_015718`. Analisi dei 12 JSON dataset e del manifest; risultati originali non modificati.

## Stato complessivo

- 72 run marcate `completed`; 48 convergenti.
- 22 arresti con avvisi e 2 arresti al limite di 25 iterazioni.
- Il riepilogo `COMPLETED` descrive il termine del programma: `DatasetResults.complete` registra anche i normali arresti con avvisi come completati. Non significa assenza di problemi tecnici o convergenza.

## Configurazione realmente eseguita

Il manifest contiene 6 configurazioni: esempi 0/1/4 a temperatura 0.2; quattro esempi a temperature 0, 0.4 e 0.6. Soglia 0.68 in tutte le run.
Non sono risultati del nuovo disegno a 5 configurazioni e 60 run (prompting a 0.3 e few-shot a 0/0.3/0.6). Non rinominare questi risultati come appartenenti al nuovo piano.

## Problemi rilevati

- Limite del contesto degli evaluator: 11 run con almeno un voto invalido per input troppo lungo. Contesto caricato: 8192 token.
- Output degli evaluator troncato: 2 run con almeno un voto invalido per `finish_reason='length'`. Budget richiesto: 4096 token. Il messaggio non distingue sempre tra esaurimento dell'output e spazio residuo nel contesto.
- Output Qwen troncato durante la riparazione: 3 run. Budget generatore: 4000 token.
- Riparazioni respinte: 5 run con almeno un rigetto; 11 raggiungono il budget di tre passaggi di riparazione.
- Le categorie possono sovrapporsi. Un voto invalido non rende necessariamente invalida la decisione: possono essere sufficienti due altri voti validi concordi. La mancata convergenza da sola non dimostra un errore tecnico.

## Dettaglio per dataset

| Dataset | Convergenti / 6 | Run con contesto insufficiente | Run con output critic troncato | Run con output Qwen troncato |
|---|---:|---:|---:|---:|
| Assegno Unico Universale - SIA Project 24 25 | 0 | 6 | 0 | 0 |
| Calculator | 6 | 0 | 0 | 0 |
| Ethical Purchasing Group - SIA Project 22 23 | 1 | 4 | 2 | 0 |
| Event Organization Portal - SIA Project 21 22 | 6 | 0 | 0 | 0 |
| Fitness Tracker | 5 | 0 | 0 | 0 |
| Genome Nexus | 6 | 0 | 0 | 0 |
| Gestao Hospital | 4 | 0 | 0 | 0 |
| La Reine Marlene - SIA Project 23 24 | 6 | 0 | 0 | 0 |
| London Ambulance Service | 4 | 0 | 0 | 0 |
| Planning Poker | 4 | 0 | 0 | 0 |
| Recycling | 4 | 0 | 0 | 0 |
| SIA Project 25 26 | 2 | 1 | 0 | 3 |

## Run al limite di iterazioni

- Ethical Purchasing Group - SIA Project 22 23 / E1_0_examples: 25 iterazioni; 9 branch inconcludenti alla fine; output critic troncato presente: True.
- Fitness Tracker / E2_temperature_0.0: 25 iterazioni; 1 branch inconcludenti alla fine; output critic troncato presente: False.

## Esempi diagnostici

- Ethical Purchasing Group, zero-shot: nell'ultima iterazione Ministral e Gemma restituiscono output incompleti, mentre Granite produce voti validi. Rimane un solo voto utilizzabile, quindi non si raggiunge il quorum; il ciclo arriva a 25 iterazioni.
- Fitness Tracker, few-shot a temperatura 0: nell'ultimo branch irrisolto i tre voti sono validi ma diversi (REWRITE, KEEP e DUPLICATE_REMOVAL). Questo arresto al limite rappresenta un mancato accordo del metodo, distinto da un errore di contesto o da una risposta troncata.
- Un voto di Gemma in Fitness Tracker e stato respinto per un superstite non appartenente ai candidati diretti dello stesso attore: il controllo di validazione ha impedito di applicare tale proposta.

`context_limit`:

```text
BadRequestError: Error code: 400 - {'error': 'The number of tokens to keep from the initial prompt is greater than the context length (n_keep: 9665>= n_ctx: 8192). Try to load the model with a larger context length, or provide a shorter input.'}
```

`critic_output_limit`:

```text
EvaluatorGenerationError: Structured critic response incomplete or unavailable (finish_reason='length'). Check output-token and loaded-context budgets; no partial vote accepted.
```

## Interpretazione e interventi

1. Risolvere i limiti di contesto prima di confrontare le metriche: il payload globale include confronti con tutti gli HLG e cresce con il numero dei branch. Per Assegno Unico le sei configurazioni si arrestano tutte alla prima iterazione con problemi di contesto.
2. Dimensionare insieme contesto caricato e budget di risposta. Aumentare il solo `max_tokens` non garantisce che input e output rientrino nel contesto. Verificare i valori effettivamente caricati in LM Studio.
3. Adeguare anche il budget Qwen per la rigenerazione delle collezioni complete; conservare il rifiuto delle risposte incomplete.
4. Rieseguire le run tecnicamente bloccate in una nuova suite con il disegno corrente e parametri di esecuzione uniformi. Conservare separatamente questa suite storica.
5. Distinguere nell'analisi stato di esecuzione, problemi tecnici, arresti del metodo, convergenza e metriche esterne. Non trattare una gerarchia invariata per errore di contesto come confermata.

Dettaglio di tutte le 72 run: `bottom_up_execution_analysis.csv`. I conteggi dei voti non sono conteggi di richieste: una risposta globale ? ripetuta nei record dei branch e i retry grezzi non sono conservati.
