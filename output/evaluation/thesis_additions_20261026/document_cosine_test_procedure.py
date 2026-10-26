"""Explain the executed cosine procedure from local code and recorded artefacts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'experiment_threhold_tuning/outputs/all_datasets_20261025_170409'
review = json.loads((RUN / 'parent_review.json').read_text(encoding='utf-8'))
summary = json.loads((RUN / 'cosine_summary.json').read_text(encoding='utf-8'))
pairs = [p for d in review['datasets'] for p in d['pairs']]
assert len(review['datasets']) == 12 and len(pairs) == 61
assert sum(p['status'] == 'completed' for p in pairs) == 60
assert all(p['human_review']['equivalent'] is None for p in pairs)
assert review['selected_threshold'] is None
assert hashlib.sha256((RUN / 'parent_review.json').read_bytes()).hexdigest() == summary['source_sha256']

body = r'''\paragraph{Come sono stati eseguiti i test della cosine similarity.}
La procedura effettivamente usata per questo esperimento \`e implementata
in \path{experiment_threhold_tuning/parent_review.py} e
\path{experiment_threhold_tuning/cosine_summary.py}. Il primo script
genera nuove ricostruzioni e i confronti col padre; il secondo riusa
quelle ricostruzioni per il riepilogo e per i confronti con gli altri
HLG. Le fasi sono le seguenti.
\begin{enumerate}
\item \textbf{Preparazione dei dati.} Sono caricate le dodici baseline
\path{*_ZERO_SHOT_feedback_original_top_down_top_down.json} da
\path{output/top_down/}. HLG e LLG sono validati con i data model;
per ogni padre vengono raccolti i figli il cui riferimento corrisponde
al padre. Il JSON registra nomi, attori, descrizioni, figli non associati
e checksum SHA-256 delle baseline e del codice. Il raggruppamento diretto
di questo test non applica il recupero dei riferimenti del ciclo completo.
\item \textbf{Ricostruzione indipendente dal padre.} Per ogni gruppo non
vuoto, Qwen riceve soltanto le descrizioni degli LLG e il prompt del
ricostruttore della pipeline. Il padre originale e il documento restano
nel file di audit, ma non vengono passati alla chiamata di ricostruzione.
Non intervengono evaluator, maggioranza o riparazione. Si ottiene una
descrizione HLG$'$ nuova per ciascuno dei 60 gruppi disponibili; il padre
senza figli rimane marcato \texttt{not\_reconstructable}.
\item \textbf{Calcolo dei punteggi.} Le descrizioni di HLG$'$ e HLG
originale sono codificate con
\path{sentence-transformers/all-mpnet-base-v2} su CPU, senza il
preprocessing NLTK usato nella valutazione BERT finale. Si usano le
funzioni \path{_encode_text} e \path{_cosine} della pipeline: controllo
della lunghezza, embedding normalizzati, verifica dei vettori e prodotto
scalare normalizzato. Il risultato di ogni coppia col padre e la sua
etichetta provvisoria a 0,68 sono salvati subito nel JSON, cos\`i che una
ripresa possa riusare una ricostruzione gi\`a registrata. L'etichetta
numerica non \`e un'annotazione umana di equivalenza.
\item \textbf{Estensione agli altri goal.} Dopo il completamento delle
ricostruzioni, il riepilogo con opzione \path{--include-cross-goal}
confronta ciascun HLG$'$ disponibile con ogni altro HLG originale del
medesimo dataset. Le coppie sono direzionali e non comprendono confronti
tra due HLG$'$ o tra dataset diversi. Si distinguono i 334 confronti
fuori diagonale e il loro sottoinsieme di 112 coppie con lo stesso attore;
il nome dell'attore viene normalizzato per spazi e maiuscole. La
candidatura duplicata a 0,68 richiede anche la ricostruibilit\`a del
ramo candidato. Il file mantiene pure i confronti che non sono candidati.
\item \textbf{Riepilogo e sensibilit\`a alla soglia.} Per i confronti col
padre, con altri HLG e con altri HLG dello stesso attore vengono calcolati
numerosit\`a, media, mediana, deviazione standard della popolazione,
minimo e massimo, globalmente e per dataset. Sono distinte la media
di tutte le coppie e la media delle medie dei dataset. La successiva
analisi di sensibilit\`a conta le stesse coppie sopra ciascuno dei valori
0,50; 0,55; 0,60; 0,65; 0,68; 0,70; 0,75; 0,80; 0,85; 0,90.
Le ricostruzioni restano fisse durante questi confronti: cambiare la
soglia non provoca una nuova generazione dei testi. La
Tabella~\ref{tab:completed-threshold-grid} riporta i conteggi e la
Figura~\ref{fig:completed-threshold-analysis} ne mostra l'andamento.
\end{enumerate}

\paragraph{File prodotti e riproducibilit\`a.}
Il run analizzato \`e \path{all_datasets_20261025_170409}.
\path{parent_review.json} contiene gli input, le ricostruzioni e i
punteggi col padre; \path{cosine_pairs.csv} contiene tutte le 394 coppie;
\path{cosine_by_dataset.csv} contiene le statistiche per dataset e
\path{dataset_coverage.csv} la copertura. Il riepilogo \`e disponibile
anche in \path{cosine_summary.json} e \path{cosine_summary.xlsx}.
Il JSON di riepilogo conserva l'hash del file sorgente e la revisione
risolta del modello embedding, per collegare le statistiche ai dati
effettivamente usati. Il riepilogo ammette soltanto dodici dataset
distinti, con rami completati o esplicitamente non ricostruibili.

I comandi seguenti descrivono come avviare una \emph{nuova} esecuzione
dalla radice del repository. Il percorso di uscita deve essere distinto
dall'archivio analizzato; una rigenerazione pu\`o produrre nuovi testi e
quindi nuovi punteggi. Il test richiede il backend locale configurato
e verifica che non vi siano esperimenti attivi sul server condiviso.
\begin{lstlisting}[style=llmprompt,caption={Comandi per una nuova esecuzione del test cosine e del riepilogo con confronti fra goal.},label={lst:cosine-test-commands}]
python -m experiment_threhold_tuning.parent_review --output experiment_threhold_tuning/outputs/nuovo_run/parent_review.json
python -m experiment_threhold_tuning.cosine_summary experiment_threhold_tuning/outputs/nuovo_run/parent_review.json --include-cross-goal
\end{lstlisting}

\paragraph{Che cosa \`e stato verificato e che cosa rimane da validare.}
L'esecuzione completa i calcoli numerici, ma non una calibrazione
supervisionata della soglia: il JSON \`e nello stato
\path{awaiting_human_review}, il campo \path{selected_threshold} \`e
nullo e nessuno dei 61 campi \path{human_review.equivalent} \`e compilato.
Non sono quindi disponibili etichette manuali per stimare veri o falsi
duplicati, un insieme di validazione annotato o un ottimo di Precision,
Recall e F1 della soglia. La ground truth dei progetti non \`e usata
in questa procedura. Il valore 0,68 resta operativo e descrittivo;
i valori medi osservati non ne costituiscono una stima automatica.

Separatamente, i test software in \path{tests/test_bottom_up_review.py}
verificano con vettori controllati la soglia condivisa, la direzionalit\`a
dei candidati, i rami senza ricostruzione e la gestione degli errori di
embedding. Questi test controllano la logica del codice; non misurano
l'accuratezza semantica di MPNet sui dodici progetti e non sostituiscono
l'annotazione delle coppie.

'''

path = ROOT / 'main.tex'
text = path.read_text(encoding='utf-8')
start, end = '% BEGIN COSINE_TEST_PROCEDURE', '% END COSINE_TEST_PROCEDURE'
block = start + '\n' + body + end + '\n\n'
if start in text:
    a = text.index(start); b = text.index(end, a) + len(end)
    text = text[:a] + block.rstrip() + text[b:]
else:
    anchor = r'\paragraph{Distribuzione dei valori osservati.}'
    assert text.count(anchor) == 1
    text = text.replace(anchor, block + anchor)
path.write_text(text, encoding='utf-8')
sources = [RUN/'parent_review.json', RUN/'cosine_summary.json',
           ROOT/'experiment_threhold_tuning/parent_review.py',
           ROOT/'experiment_threhold_tuning/cosine_summary.py',
           ROOT/'src/bottom_up/goal_similarity.py', ROOT/'tests/test_bottom_up_review.py']
(OUT/'cosine_test_procedure_evidence.json').write_text(json.dumps({
    'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    'verified': {'datasets':12,'original_parents':61,'completed_reconstructions':60,'manual_labels':0,
                 'selected_threshold':None,'summary_hash_matches':True},
    'new_inference_calls':0,'human_author_review':'pending'}, indent=2),encoding='utf-8')
print('Added executed cosine-test procedure, reproducibility commands and annotation-status evidence.')
