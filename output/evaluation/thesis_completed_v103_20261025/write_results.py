"""Bind the Results chapter to the frozen 36-run analysis and real trace excerpts."""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
snapshot = json.loads((OUT / "source_snapshot.json").read_text(encoding="utf-8"))
def read(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

comparison = {r["Esperimento"]: r for r in read("final_comparison.csv")}
details = read("dataset_comparison.csv")
operational = {r["config_id"]: r for r in read("operational_summary.csv")}
runs = {(r["dataset"], r["config_id"]): r for r in read("iteration_counts.csv")}
names = {"reference": "P4", "E1_0_examples": "P0", "E1_1_examples": "P1"}
short = {"Assegno Unico Universale - SIA Project 24 25": "AUU",
         "Calculator": "Calculator", "Ethical Purchasing Group - SIA Project 22 23": "Ethical Purchasing",
         "Event Organization Portal - SIA Project 21 22": "Event Organization",
         "Fitness Tracker": "Fitness Tracker", "Genome Nexus": "Genome Nexus",
         "Gestao Hospital": "Gestao Hospital", "La Reine Marlene - SIA Project 23 24": "La Reine Marlene",
         "London Ambulance Service": "London Ambulance", "Planning Poker": "Planning Poker",
         "Recycling": "Recycling", "SIA Project 25 26": "SIA 25--26"}
order = [d["name"] for d in snapshot["documents"]]
def number(value, signed=False):
    return (f"{float(value):+.4f}" if signed else f"{float(value):.4f}").replace('.', ',')

def metric_table(level):
    rows = []
    for config in ("Top-down originale", "E1_0_examples", "E1_1_examples", "reference"):
        row = comparison[config]
        rows.append(" & ".join([names.get(config, "Top-down"), *[number(row[f"{level} {m}"]) for m in ("Precision", "Recall", "F1")], number(row[f"{level} Delta F1"], True)]) + r" \\")
    return r"""\begin{table}[htbp]
\centering
\caption{Metriche semantiche macro degli LEVEL su tutti i 12 dataset. Ogni riga bottom-up comprende 12 run completate, incluse quelle non convergenti.}
\label{tab:results-LEVEL-macro}
\begin{tabular}{lrrrr}
\toprule
\textbf{Metodo} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & $\boldsymbol{\Delta F1}$ \\
\midrule
ROWS
\bottomrule
\end{tabular}
\end{table}
""".replace("LEVEL", level).replace("ROWS", '\n'.join(rows))

def dataset_table(config):
    rows = []
    for dataset in order:
        row = next(r for r in details if r["Esperimento"] == config and r["Dataset"] == dataset)
        run = runs[(dataset, config)]
        iterations = run["iterations"] + (r"$^{*}$" if run["converged"] == "False" else "")
        rows.append(" & ".join([short[dataset], number(row["HLG F1"]), number(row["HLG Delta F1"], True), number(row["LLG F1"]), number(row["LLG Delta F1"], True), iterations]) + r" \\")
    return r"""\begin{table}[htbp]
\centering
\small
\caption{Risultati di CONFIG per dataset. I delta sono differenze assolute rispetto alla stessa baseline top-down. L'asterisco indica arresto a 25 iterazioni senza convergenza.}
\label{tab:results-CONFIG-datasets}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Dataset} & \textbf{HLG F1} & $\boldsymbol{\Delta F1_H}$ & \textbf{LLG F1} & $\boldsymbol{\Delta F1_L}$ & \textbf{Iter.} \\
\midrule
ROWS
\bottomrule
\end{tabular}
\end{table}
""".replace("CONFIG", names[config]).replace("ROWS", '\n'.join(rows))

def locate(dataset, config, iteration, branch):
    doc = next(d for d in snapshot["documents"] if d["name"] == dataset)
    trace = doc["runs"][config + "/repeat_001"]["globalGoalCycle"]["iterations"][iteration - 1]
    return doc, trace, trace["input_hierarchy"]["high_level_goals"]["goals"][int(branch) - 1]

event = "Event Organization Portal - SIA Project 21 22"
doc_event, trace_event, parent_event = locate(event, "E1_0_examples", 1, "1")
event_vote = next(v for v in trace_event["evaluator_responses"] if v["model"] == "mistralai/ministral-3-14b-reasoning")
event_decision = next(d for d in event_vote["decisions"] if d["branch_id"] == "1")
_, trace_fitness, parent_fitness = locate("Fitness Tracker", "reference", 25, "2")
fitness = [(v["model"], next(d for d in v["decisions"] if d["branch_id"] == "2"))
           for v in trace_fitness["evaluator_responses"] if v["model"] != "google/gemma-4-12b-qat"]
doc_auu, trace_auu, parent_auu = locate(order[0], "E1_1_examples", 1, "8")
auu_vote = next(d for d in trace_auu["evaluator_responses"][0]["decisions"] if d["branch_id"] == "8")

def listing(decision, label, caption):
    return (r"\begin{lstlisting}[style=llmjson,caption={" + caption + r"},label={" + label + "}]\n"
            + json.dumps(decision, ensure_ascii=True, indent=2) + "\n" + r"\end{lstlisting}" + "\n")

evidence = {
    "E1_event_rewrite": {"dataset": event, "config_id": "E1_0_examples", "iteration": 1,
        "branch_id": "1", "parent": parent_event, "reconstruction": trace_event["reconstructions"]["1"],
        "model": event_vote["model"], "decision": event_decision},
    "E2_fitness_conflict": {"dataset": "Fitness Tracker", "config_id": "reference", "iteration": 25,
        "branch_id": "2", "parent": parent_fitness, "reconstruction": trace_fitness["reconstructions"]["2"],
        "decisions": fitness},
    "E3_auu_duplicate": {"dataset": order[0], "config_id": "E1_1_examples", "iteration": 1,
        "branch_id": "8", "removed_parent": parent_auu,
        "survivor": trace_auu["input_hierarchy"]["high_level_goals"]["goals"][2],
        "decision": auu_vote, "outcome": trace_auu["structural_action_outcomes"]["8"]}}
(OUT / "case_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

chapter = r"""\chapter{Results}
\label{ch:results}

\section{Esecuzioni incluse e provenienza dei risultati}
% Evidence: evidence_manifest.json; source_snapshot.json, configurations/documents.
Questo capitolo riporta i risultati della suite
\path{ofat_fixed_thresholds_20261024_211627}, limitatamente alle configurazioni
completate su tutti e dodici i dataset: P0, zero-shot; P1, one-shot;
P4, few-shot con quattro esempi, identificata come \texttt{reference} nei file.
La temperatura degli evaluator \`e 0,3 e la soglia MPNet \`e 0,68 in tutti
e tre i casi. L'analisi comprende quindi \textbf{36 run bottom-up}, una
per coppia dataset--configurazione, confrontate con le stesse dodici
baseline top-down. Non sono incluse medie delle configurazioni di
temperatura T0 e T6, che non erano complete sull'intero corpus al momento
dello snapshot. RQ2 viene pertanto affrontata qui soltanto per il prompting.

Una run completata viene inclusa anche se termina al limite di 25
iterazioni senza convergere. Non vengono selezionati soltanto gli esiti
favorevoli, n\'e usati checkpoint intermedi come risultati finali.
La suite registra la policy
\path{cosine_global_documentary_review_v103_conditional_parent_reference_recovery_schema3}.
Le modifiche successive al codice non descrivono retroattivamente queste
esecuzioni. Lo snapshot dei dati, le metriche, i conteggi e gli estratti
dei casi discussi sono conservati in
\path{output/evaluation/thesis_completed_v103_20261025/}.
\texttt{evidence\_manifest.json} registra provenienza e checksum;
\texttt{case\_evidence.json} identifica dataset, configurazione,
iterazione, ramo e modello per ciascun estratto.

Le metriche sono state calcolate nuovamente sugli output finali con
\texttt{bert-base-uncased}, preprocessing NLTK e matching uno-a-uno
descritti nel Capitolo~\ref{ch:experiments}. Precision, Recall e F1 sono
mediate separatamente sui dodici dataset: ogni progetto ha lo stesso
peso, indipendentemente dal numero dei suoi goal. I delta sono differenze
assolute rispetto alla baseline appaiata, non percentuali. L'analisi
descrive una sola ripetizione per configurazione e non stima la
variabilit\`a tra generazioni o una significativit\`a statistica.

\section{Confronto complessivo con il top-down}
% Evidence: final_comparison.csv, all four rows; final_metrics.csv, 96 rows.
HLG_TABLE
LLG_TABLE

Tutte e tre le configurazioni aumentano l'F1 medio degli HLG rispetto
al top-down: da 0,5297 a 0,5372 con P4, 0,5418 con P0 e 0,5428 con P1.
Il vantaggio di P1 su P0 \`e circa 0,0010 e va letto come differenza
descrittiva della singola campagna. Gli incrementi di Precision non
coincidono sempre con incrementi di Recall: P0 passa da 0,5794 a 0,6047
in Precision, ma da 0,5725 a 0,5537 in Recall. Una maggiore selettivit\`a
o una diversa formulazione dei goal pu\`o quindi migliorare il matching
senza aumentare la copertura misurata dei riferimenti.

Per gli LLG, P0 ottiene l'incremento maggiore: F1 da 0,5083 a 0,5558,
con delta +0,0475. La variazione accompagna soprattutto la Recall,
che passa da 0,4675 a 0,5135, mentre la Precision cambia da 0,6866
a 0,6883. P4 raggiunge F1 0,5153, con delta +0,0069; P1 rimane
sostanzialmente sul valore iniziale, con differenza non arrotondata
pari a circa $-0{,}000073$. Non emerge quindi una configurazione
uniformemente migliore su entrambi i livelli: P1 ha il maggiore F1
HLG, mentre P0 ha il maggiore F1 LLG.

Questi risultati non dimostrano che introdurre pi\`u esempi migliori
le prestazioni. P1 contiene un unico scenario con decisioni di
mantenimento; P4 aggiunge anche dimostrazioni di riscrittura e rimozione.
Il confronto modifica quindi contenuto, copertura delle azioni e
lunghezza del prompt, oltre al numero di esempi. Le differenze
osservate riguardano queste specifiche configurazioni.

\section{Risultati per dataset e configurazione}
% Evidence: dataset_comparison.csv; iteration_counts.csv.
Le Tabelle~\ref{tab:results-P0-datasets}--\ref{tab:results-P4-datasets}
separano i risultati delle tre configurazioni. AUU indica Assegno
Unico Universale; gli altri nomi abbreviati identificano i progetti
elencati nel corpus. I conteggi di iterazioni descrivono l'esito
operativo, mentre F1 descrive il matching con la ground truth.

P0_TABLE
P1_TABLE
P4_TABLE

Gli incrementi aggregati nascondono comportamenti eterogenei.
Con P0, l'F1 degli HLG aumenta in tre dataset e diminuisce in sei,
restando invariato negli altri tre; i guadagni pi\`u ampi di AUU,
La Reine Marlene e Recycling compensano le diminuzioni.
Per gli LLG della stessa configurazione si osservano sette incrementi,
due diminuzioni e tre valori invariati. Con P1, invece, sette dataset
mantengono sia il valore HLG sia quello LLG della baseline: il maggiore
F1 HLG medio non implica una revisione estesa di tutti i progetti.

Recycling mostra il disaccoppiamento tra livelli: con P4 l'F1 HLG
passa da 0,6311 a 0,7586, mentre l'F1 LLG scende da 0,5906 a 0,5116.
Il miglioramento del padre non garantisce quindi una decomposizione
migliore. Anche London Ambulance Service converge con P4 dopo tredici
iterazioni, ma l'F1 LLG diminuisce da 0,4841 a 0,2877.
Viceversa, in Event Organization P0 aumenta l'F1 LLG da 0,3855 a
0,6242, pur riducendo l'F1 HLG da 0,5532 a 0,5265.
Le due misure devono essere interpretate separatamente.

\section{Convergenza, decisioni e costo iterativo}
% Evidence: operational_summary.csv; iteration_counts.csv.
OPERATIONAL_TABLE

P1 converge su tutti i dodici dataset in 37 iterazioni globali
complessive; P0 converge su undici dataset e ne richiede 105; P4
converge su dieci e ne richiede 106. Le medie sono rispettivamente
3,08, 8,75 e 8,83 iterazioni per dataset. Questi conteggi indicano
lavoro iterativo e non tempi di esecuzione: la durata dipende anche
da dimensione dei prompt, tentativi locali e caricamento dei modelli.
AUU non converge con P0 entro 25 iterazioni; Fitness Tracker e
SIA 25--26 non convergono con P4. SIA 25--26 converge invece
all'ultima iterazione consentita con P0.

DECISIONS_TABLE

Le decisioni della Tabella~\ref{tab:results-actions} sono conteggi
di rami esaminati attraverso tutte le iterazioni, non di goal distinti.
Una riscrittura approvata pu\`o produrre un passaggio di riparazione
rifiutato; pi\`u rami riscritti nella stessa iterazione condividono
un solo passaggio top-down. Per questo i 232 voti aggregati di
riscrittura di P4 corrispondono a 86 passaggi accettati e otto
rifiutati, non a 232 chiamate indipendenti alla pipeline.

Nell'ultima iterazione delle tre run non convergenti, AUU con P0
mantiene tre rami e richiede quattro riscritture; Fitness Tracker
con P4 mantiene due rami, richiede due riscritture e lascia un ramo
inconcludente; SIA 25--26 con P4 mantiene cinque rami e richiede tre
riscritture. Le altre 33 run terminano con tutti i rami confermati.
Questi esiti conclusivi sono distinti dai totali dell'intero percorso
nella Tabella~\ref{tab:results-actions}.

Gli evaluator forniscono 318 risposte globali valide in P4, 312 valide
e tre non valide in P0, 110 valide e una non valida in P1. Questi
conteggi riguardano l'esito finale del budget di tentativi di ciascun
modello per iterazione, non tutti i tentativi tecnici. La maggior parte
degli output rispetta dunque il contratto JSON; la validit\`a
strutturale non garantisce per\`o la compatibilit\`a delle critiche
o la correttezza documentale della decisione.

MODEL_TABLE

I modelli non utilizzano le azioni con la stessa frequenza.
In P1, Granite propone 179 mantenimenti e cinque riscritture,
Ministral 127 mantenimenti e 53 riscritture, Gemma 103 mantenimenti
e 72 riscritture. La stessa informazione non conduce quindi a
giudizi uniformi. Questi conteggi non misurano l'accuratezza dei
singoli evaluator: descrivono il loro comportamento sulle gerarchie
effettivamente visitate. I confronti tra configurazioni dipendono
anche dal numero diverso di iterazioni e rami esaminati. Il dettaglio
per modello, dataset e configurazione, insieme alle decisioni aggregate
dell'intera run e dell'ultima iterazione, rimane nei CSV dell'analisi.

\section{Casi reali: contributi utili e limiti}
Gli esempi seguenti sono estratti dagli output effettivamente salvati,
non dalle dimostrazioni few-shot. I listing riproducono la singola
decisione selezionata dentro una risposta globale pi\`u ampia.
Il numero del ramo vale esclusivamente nello snapshot indicato.

\subsection{Recupero di capacit\`a documentate: Event Organization, P0}
% Evidence: case_evidence.json/E1_event_rewrite; source_snapshot.json, Event/P0;
% final_metrics.csv, Event Organization, top_down and E1_0_examples, LLG.
Alla prima iterazione, il ramo 1 ha come padre
\emph{Manage Event Planning and Coordination}, la cui descrizione
menziona inviti, prenotazione dei servizi e budget. La ricostruzione
\`e \emph{Manage event planning and financial oversight from setup
to execution}. Il documento fornito ai modelli descrive anche la
richiesta di appuntamenti ai fornitori e l'accettazione o il rifiuto
dei preventivi, capacit\`a non esplicite nella descrizione iniziale
del padre. Ministral produce l'estratto seguente; Gemma propone
anch'esso REWRITE, mentre Granite propone KEEP.

EVENT_JSON

Nella gerarchia finale compaiono, tra gli altri, i goal
\emph{Manage Quotations and Approvals}, descritto come la possibilit\`a
di visualizzare, approvare o rifiutare preventivi, e
\emph{Manage Payment Schedules for Event Services}, che include
rate, anticipo e saldo. Entrambi hanno un riscontro esplicito nella
descrizione del progetto usata dalla pipeline.
Gli LLG passano da 10 a 22 su 28 riferimenti; la Recall semantica
sale da 0,2616 a 0,5573 e la Precision diminuisce da 0,7325 a 0,7093.
Il guadagno F1 osservato \`e quindi coerente con una maggiore copertura
testuale accompagnata da una lieve perdita di Precision. La
riscrittura riapre la decomposizione e rende possibile tale recupero;
il caso non isola il contributo causale di HLG$'$ rispetto alla
rigenerazione completa degli LLG e agli altri segnali disponibili.

\subsection{Conservazione della baseline: Calculator}
% Evidence: source_snapshot.json, Calculator, all selected configurations;
% final_metrics.csv, Calculator; iteration_counts.csv.
Calculator termina alla prima iterazione in tutte e tre le
configurazioni, confermando l'unico HLG
\emph{Enable quick and accurate basic arithmetic operations}.
La collezione rimane composta da un HLG e cinque LLG, con F1 HLG
0,5366 e F1 LLG 0,6083 invariati. Il meccanismo evita una rigenerazione
in questo caso, ma l'accordo dei modelli non rende perfetta la
baseline rispetto ai riferimenti. La conservazione costituisce un
esito utile di stabilit\`a, distinto da un miglioramento delle metriche.

\subsection{Critiche incompatibili: Fitness Tracker, P4}
% Evidence: case_evidence.json/E2_fitness_conflict.
All'iterazione 25, il ramo 2 riguarda
\emph{Support users in setting and achieving personalized fitness goals}.
Il padre consente gi\`a di definire obiettivi misurabili con scadenze,
usare dati storici, monitorare il progresso e ricevere feedback e
promemoria. HLG$'$ sintetizza obiettivi definiti, visualizzazione dei
progressi e promemoria. Ministral e Granite approvano entrambi
REWRITE, ma forniscono istruzioni divergenti:

FITNESS_MINISTRAL

FITNESS_GRANITE

La prima critica esclude la definizione di obiettivi e scadenze;
la seconda chiede di esplicitare l'uso dei dati storici per definire
gli obiettivi, gi\`a presente nel padre corrente. Gemma propone KEEP.
I due voti di riscrittura raggiungono il quorum, ma non esprimono
accordo sul contenuto della correzione. La concatenazione trasferisce
entrambe le istruzioni al generatore. Questa incompatibilit\`a
fornisce una spiegazione plausibile delle revisioni persistenti,
senza dimostrare da sola la causa di tutte le oscillazioni.
La run termina senza convergenza, con due riscritture e un ramo
inconcludente nell'ultimo snapshot; gli incrementi F1 rispetto alla
baseline sono soltanto +0,0069 per gli HLG e +0,0050 per gli LLG.

\subsection{Una duplicazione approvata non garantisce copertura: AUU, P1}
% Evidence: case_evidence.json/E3_auu_duplicate; AUU baseline description and P1 final goals.
Alla prima iterazione, tutti e tre gli evaluator chiedono di eliminare
il ramo 8, \emph{Support application modifications and additions after
submission}, come duplicato del ramo 3,
\emph{Enable real-time fiscal code validation and conflict detection}.
L'output di Ministral \`e il seguente; gli altri due modelli indicano
la stessa coppia di identificativi.

AUU_JSON

Il primo padre descrive l'aggiunta di figli a domande esistenti e la
modifica dei dati di pagamento o delle dichiarazioni; il superstite
descrive la validazione dei codici fiscali e il blocco delle domande
in conflitto. I testi condividono una parte delle funzionalit\`a,
ma il secondo non esplicita tutte quelle del primo. La descrizione
del progetto richiede espressamente di aggiungere figli a una domanda
esistente e confermare o modificare i dati gi\`a presenti.
Il codice applica \texttt{REMOVED\_WITH\_CHILDREN}; negli HLG finali
non compare pi\`u una formulazione esplicita dell'aggiunta di figli.
Questo \`e un caso in cui l'unanimit\`a non dimostra la copertura
completa necessaria alla rimozione. L'analisi riguarda il testo
conservato e non costituisce una verifica formale di tutta la gerarchia.
La run converge in quattro iterazioni e migliora F1 HLG di +0,0341,
ma riduce F1 LLG di $-0{,}0614$: convergenza e miglioramento del
matching degli HLG possono coesistere con una perdita agli altri livelli.

\subsection{Granularit\`a e dettagli non supportati: London e Recycling, P4}
% Evidence: source_snapshot.json, London/reference and Recycling/reference final LLGs;
% final_metrics.csv, respective paired baseline and reference rows.
In London Ambulance Service gli LLG passano da sei a dodici, a fronte
di tre riferimenti. La Precision scende da 0,3631 a 0,1798, mentre
la Recall passa da 0,7262 a 0,7194. Poich\'e il matching \`e uno-a-uno,
al massimo tre goal generati contribuiscono alla somma assegnata;
gli altri aumentano il denominatore della Precision. La crescita
del numero dei goal non si traduce quindi in maggiore copertura
misurata. Tra gli output finali compare
\emph{Load real-time traffic data for current route}, mentre la
descrizione fornita richiede il raggiungimento entro undici minuti,
l'allocazione dei veicoli e l'aderenza a percorsi standard, senza
specificare un servizio di traffico in tempo reale.

In Recycling la descrizione richiede sicurezza dei dati in termini
generali, ma la decomposizione finale contiene
\emph{System uses AES-256 encryption for data at rest and TLS 1.3
for data in transit}. Queste specifiche tecniche non compaiono nel
testo del progetto usato dai modelli. Inoltre, tra gli LLG finali
non sono espliciti la mappa e le notifiche degli eventi, citate nella
descrizione. Il miglioramento F1 HLG di +0,1275 non equivale pertanto
a una correzione completa della gerarchia. La revisione usa le sintesi
HLG$'$ e non i testi integrali degli LLG: questo limita la visibilit\`a
dei dettagli introdotti o omessi durante la decomposizione.

\section{Interpretazione rispetto alle domande di ricerca}
Per RQ1, l'estensione produce incrementi medi di F1 HLG in tutte
le configurazioni complete e incrementi F1 LLG in P0 e P4, mentre
P1 mantiene sostanzialmente il valore LLG iniziale. I risultati per
dataset mostrano miglioramenti, conservazioni e peggioramenti:
l'effetto del feedback dipende dal progetto e dal livello considerato.
Il meccanismo funziona in modo utile quando conserva una gerarchia
senza richiedere revisioni o quando una critica documentata riapre
una decomposizione incompleta. Mostra invece limiti nei conflitti
tra critiche, nelle rimozioni di funzionalit\`a complementari e nelle
decomposizioni che introducono granularit\`a o dettagli non supportati.

Per la parte di RQ2 dedicata al prompting, P1 presenta la maggiore
stabilit\`a operativa e P0 il maggiore F1 LLG. Il minor numero di
revisioni con P1 \`e compatibile con un comportamento pi\`u conservativo
e con il contenuto del suo esempio, che mostra mantenimenti; questa
\`e un'interpretazione, non un effetto causale isolato dal disegno.
I risultati non sostengono una superiorit\`a generale del few-shot.
Non viene formulata una conclusione sulla temperatura, poich\'e
T0 e T6 non fanno parte del confronto completo qui congelato.

Le spiegazioni restano limitate da una sola ripetizione, dal matching
testuale che non valuta direttamente attori e associazioni padre--figlio,
e dal riuso della generazione top-down dell'intera collezione durante
le riparazioni. La convergenza misura l'accordo operativo finale,
non la correttezza dei requisiti. Questi limiti spiegano perch\'e
occorre leggere metriche e tracce insieme e perch\'e un singolo F1
aggregato non basta a stabilire dove la gerarchia sia stata corretta.

"""

ops_rows = []
action_rows = []
for config in ("E1_0_examples", "E1_1_examples", "reference"):
    row = operational[config]
    ops_rows.append(" & ".join([names[config], row["converged"] + "/12", row["total_iterations"],
        f"{float(row['mean_iterations']):.2f}".replace('.', ','), row["accepted_repair_passes"],
        row["rejected_repair_passes"], row["removed_branches"]]) + r" \\")
    action_rows.append(" & ".join([names[config], *[row[k] for k in ("keep", "rewrite", "remove", "duplicate_removal", "inconclusive")]]) + r" \\")

ops_table = r"""\begin{table}[htbp]
\centering
\small
\caption{Esiti operativi delle 36 run: convergenza, cicli globali e passaggi di riparazione della collezione. Le rimozioni contano rami effettivamente eliminati, anche se altri goal sono generati in seguito.}
\label{tab:results-convergence}
\begin{tabular}{lrrrrrr}
\toprule
 & \textbf{Conv.} & \textbf{Cicli} & \textbf{Media} & \textbf{Rip. acc.} & \textbf{Rip. rif.} & \textbf{Rimossi} \\
\midrule
ROWS
\bottomrule
\end{tabular}
\end{table}
""".replace("ROWS", '\n'.join(ops_rows))
actions_table = r"""\begin{table}[htbp]
\centering
\small
\caption{Decisioni aggregate per ramo, sommate su tutte le iterazioni. DUP indica DUPLICATE\_REMOVAL; INC indica un esito inconcludente del software, non un'azione LLM.}
\label{tab:results-actions}
\begin{tabular}{lrrrrr}
\toprule
 & \textbf{KEEP} & \textbf{REWRITE} & \textbf{REMOVE} & \textbf{DUP} & \textbf{INC} \\
\midrule
ROWS
\bottomrule
\end{tabular}
\end{table}
""".replace("ROWS", '\n'.join(action_rows))
model_counts = {}
for row in read("evaluator_decision_counts.csv"):
    key = (row["Esperimento"], row["LLM"])
    counts = model_counts.setdefault(key, {field: 0 for field in (
        "KEEP", "REWRITE", "REMOVE", "DUPLICATE_REMOVAL",
        "Risposte globali valide", "Risposte globali non valide")})
    for field in counts:
        counts[field] += int(row[field])
model_rows = []
for config in ("E1_0_examples", "E1_1_examples", "reference"):
    for model, label in (("mistralai/ministral-3-14b-reasoning", "Ministral"),
                         ("ibm/granite-3.2-8b", "Granite"), ("google/gemma-4-12b-qat", "Gemma")):
        counts = model_counts[(config, model)]
        model_rows.append(" & ".join([names[config], label, *[str(value) for value in counts.values()]]) + r" \\")
model_table = r"""\begin{table}[htbp]
\centering
\scriptsize
\caption{Proposte individuali nelle risposte globali valide, sommate sui dodici dataset. Le ultime colonne contano risposte globali valide (V) e non valide (NV), non decisioni per ramo.}
\label{tab:results-model-actions}
\begin{tabular}{llrrrrrr}
\toprule
 & \textbf{Modello} & \textbf{KEEP} & \textbf{REWRITE} & \textbf{REMOVE} & \textbf{DUP} & \textbf{V} & \textbf{NV} \\
\midrule
ROWS
\bottomrule
\end{tabular}
\end{table}
""".replace("ROWS", '\n'.join(model_rows))
for token, text in {
    "HLG_TABLE": metric_table("HLG"), "LLG_TABLE": metric_table("LLG"),
    "P0_TABLE": dataset_table("E1_0_examples"), "P1_TABLE": dataset_table("E1_1_examples"),
    "P4_TABLE": dataset_table("reference"), "OPERATIONAL_TABLE": ops_table, "DECISIONS_TABLE": actions_table,
    "MODEL_TABLE": model_table,
    "EVENT_JSON": listing(event_decision, "lst:results-event-rewrite", "Decisione reale di Ministral: Event Organization, P0, iterazione 1, ramo 1."),
    "FITNESS_MINISTRAL": listing(fitness[0][1], "lst:results-fitness-ministral", "Decisione reale di Ministral: Fitness Tracker, P4, iterazione 25, ramo 2."),
    "FITNESS_GRANITE": listing(fitness[1][1], "lst:results-fitness-granite", "Decisione reale di Granite sullo stesso ramo e snapshot."),
    "AUU_JSON": listing(auu_vote, "lst:results-auu-duplicate", "Decisione reale di Ministral: AUU, P1, iterazione 1, ramo 8; voto unanime dei tre evaluator.")}.items():
    chapter = chapter.replace(token, text)

main = ROOT / "main.tex"
text = main.read_text(encoding="utf-8")
start = text.index(r"\chapter{Results}")
end = text.index(r"\chapter{Conclusion and Future Work}", start)
text = text[:start] + chapter + text[end:]
old = ("studio delle cinque configurazioni previste. L'entità del contributo\n"
       "all'accuratezza di estrazione viene determinata dai risultati sperimentali.")
new = ("studio delle cinque configurazioni previste. Il confronto completo delle tre\n"
       "strategie di prompting mostra incrementi medi di F1 degli HLG, ma risultati\n"
       "eterogenei per gli LLG e per i singoli progetti. Nella campagna analizzata,\n"
       "zero-shot ottiene il maggiore F1 LLG medio e one-shot la maggiore stabilità\n"
       "operativa. Le configurazioni di temperatura ancora incomplete non sostengono\n"
       "una conclusione comparativa. I casi osservati mostrano inoltre che accordo\n"
       "degli evaluator e convergenza non garantiscono la correttezza della gerarchia.")
assert old in text or new in text
text = text.replace(old, new, 1)
main.write_text(text, encoding="utf-8")
(OUT / "results_chapter.tex").write_text(chapter, encoding="utf-8")
print("Updated Results with 36 runs, 12 paired baselines and four real JSON excerpts.")
