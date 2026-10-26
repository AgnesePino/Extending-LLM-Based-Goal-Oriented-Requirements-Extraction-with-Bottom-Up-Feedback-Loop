"""Document current evaluator demonstrations and implementation trade-offs locally."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

RATIONALE = r"""\section{Motivazioni delle scelte implementative e dei modelli locali}
\label{sec:design-rationale}

Le scelte descritte di seguito motivano l'organizzazione del codice
corrente. Gli identificativi dei modelli provengono da
\path{src/llm_clients.py} e sono valori predefiniti modificabili tramite
variabili d'ambiente: per ricostruire una specifica esecuzione occorre
consultarne la configurazione salvata. La loro presenza nell'implementazione
non dimostra che siano i modelli migliori per il compito; la campagna
su esempi e temperatura non costituisce un confronto tra architetture.

\subsection{Ruoli dei modelli e ragioni dell'esecuzione locale}
\label{sec:local-model-rationale}

\begin{table}[htbp]
\centering
\small
\caption{Modelli predefiniti nel codice corrente e responsabilit\`a nella pipeline.}
\label{tab:local-model-roles}
\begin{tabularx}{\textwidth}{@{}p{0.45\textwidth}X@{}}
\toprule
\textbf{Identificativo configurato} & \textbf{Ruolo implementato} \\
\midrule
\path{qwen/qwen3-vl-8b} & Generazione top-down, ricostruzione HLG$'$ e generazione delle revisioni approvate. \\
\addlinespace
\path{mistralai/ministral-3-14b-reasoning} & Critic top-down e della riparazione; primo evaluator bottom-up. \\
\addlinespace
\path{ibm/granite-3.2-8b} & Secondo evaluator bottom-up. \\
\addlinespace
\path{google/gemma-4-12b-qat} & Terzo evaluator bottom-up. \\
\addlinespace
\path{sentence-transformers/all-mpnet-base-v2} & Embedding delle descrizioni per lo screening semantico, eseguito su CPU. \\
\bottomrule
\end{tabularx}
\end{table}

L'esecuzione tramite LM Studio, con endpoint predefinito
\path{http://127.0.0.1:1234/v1}, consente al codice di indirizzare
esplicitamente ogni chiamata a un modello e di gestirne caricamento,
contesto e rilascio. Questa organizzazione rende controllabile la
successione delle chiamate nelle esecuzioni ripetute: i messaggi, il
modello richiesto e i parametri sono costruiti dalla stessa pipeline.
Con tale endpoint l'inferenza generativa avviene presso il server locale;
l'eventuale acquisizione iniziale dei pesi e delle dipendenze rimane
un'operazione distinta. L'endpoint \`e configurabile, quindi la propriet\`a
di esecuzione locale dipende anche dalla configurazione effettiva.
Non sono dedotte da questa scelta garanzie di determinismo, assenza
di costi o maggiore accuratezza rispetto a servizi esterni.

Qwen viene riutilizzato per generare e ricostruire goal attraverso
un unico client e schemi di dati condivisi. Ci\`o mantiene uniforme
l'interfaccia delle fasi generative e ne concentra la configurazione.
Le chiamate implementate forniscono testo: non viene attribuito un
vantaggio sperimentale alle capacit\`a multimodali suggerite dal nome
del modello. Il riuso dello stesso generatore comporta inoltre una
possibile dipendenza tra gli errori della decomposizione e quelli
della ricostruzione; HLG$'$ richiede perci\`o verifica documentale.

Ministral svolge il controllo top-down e partecipa anche alla revisione
bottom-up; Granite e Gemma aggiungono due giudizi prodotti da modelli
distinti. La motivazione dell'insieme eterogeneo \`e evitare che ogni
azione dipenda da una sola risposta. I tre evaluator ricevono lo stesso
snapshot, le stesse istruzioni e gli stessi esempi, senza ricevere i
voti degli altri. Si tratta di separazione delle chiamate: non prova
l'indipendenza statistica degli errori. Non essendo previsto un confronto
controllato fra questi modelli o fra uno e tre evaluator, non si pu\`o
attribuire all'eterogeneit\`a un miglioramento misurato n\'e giustificare
una graduatoria di qualit\`a tra le famiglie scelte.

Il caricamento sequenziale con \texttt{loaded\_local\_model} limita
la compresenza dei modelli della pipeline; il lock rientrante coordina
le chiamate nello stesso processo. Gli embedding su CPU lasciano
disponibili le risorse GPU alle fasi generative. Sono scelte di gestione
delle risorse, non misure di memoria o di velocit\`a: il lock non coordina
processi Python diversi e non viene assunto un requisito hardware
dimostrato dai soli identificativi dei modelli.

\subsection{Motivazioni delle regole e del coordinamento}

\paragraph{Separare screening e giudizio documentale.}
Gli embedding forniscono un criterio numerico comune per confrontare
le descrizioni; gli evaluator ricevono etichette gi\`a assegnate e
controllano il significato rispetto ai requisiti. La soglia comune
0,68 \`e una scelta operativa accompagnata dall'analisi descrittiva
dei punteggi, non un ottimo validato su coppie annotate indipendentemente.
Fornire soltanto le coppie candidate HLG$'$--HLG evita di richiedere
al modello di interpretare una matrice numerica. Gli identificativi
conservano il collegamento ai rami, mentre il vincolo sullo stesso attore
circoscrive la rimozione di duplicati. Un candidato resta da verificare;
un duplicato escluso dallo screening pu\`o invece non essere esaminato
come tale. I confronti numerici completi restano negli artefatti diagnostici.

\paragraph{Snapshot globale e conservazione delle funzionalit\`a.}
La revisione congiunta permette di considerare i confini degli altri
HLG prima di proporre riscritture o rimozioni. Leggere nome e descrizione
del padre evita di inferire omissioni da un titolo sintetico.
La ricostruzione riassume i figli senza trasmetterne tutti i testi
agli evaluator: contiene il volume del contesto, ma rende indiretta
la verifica della decomposizione. Le quattro azioni distinguono un
padre valido, uno correggibile, una funzione non supportata e una
ridondanza completa. Preservare le parti supportate e richiedere
un superstite valido impedisce, sul piano delle istruzioni, di trattare
una sovrapposizione parziale come motivo sufficiente di cancellazione.
L'interpretazione documentale resta affidata al modello: la validazione
software non dimostra da sola la correttezza semantica del giudizio.

\paragraph{Formato strutturato, quorum e feedback.}
Il JSON e i controlli sugli identificativi rendono le proposte
interpretabili dal codice. La maggioranza di almeno due voti validi
richiede accordo sull'azione e, per i duplicati, sul superstite.
Se manca il quorum, l'esito resta inconcludente: il sistema non forza
una modifica. Solo le critiche dei voti concordi sono unite e inoltrate
al generatore, cos\`i che la riparazione segua la decisione approvata.
Il feedback specifica difetto, modifica e funzionalit\`a da preservare;
il limite indicativo di lunghezza contiene il contesto senza sostituire
lo schema di validazione. Questi meccanismi controllano l'applicazione
delle proposte, ma non garantiscono che una maggioranza abbia ragione.

\paragraph{Orchestrazione, riparazione e tracciabilit\`a.}
L'orchestratore rende esplicito l'ordine ricostruzione--screening--voto--
aggiornamento e centralizza stato, callback e checkpoint.
La riparazione riusa il percorso top-down per mantenere coerenti
le strutture HLG e LLG e applica insieme le nuove collezioni solo dopo
l'approvazione del critic HLG e la validazione strutturale dei figli.
L'atomicit\`a riguarda tale sostituzione: le rimozioni gi\`a approvate
sono applicate prima e non vengono annullate da un successivo fallimento
della riparazione. I tentativi tecnici limitati e il tetto di 25 iterazioni
contengono il lavoro in caso di errori o oscillazioni; raggiungere il
tetto non equivale a convergenza. Configurazioni salvate, log e checkpoint
permettono di ricostruire l'esecuzione, senza trasformare un controllo
operativo in una prova di correttezza dei requisiti.

\paragraph{Isolare le variabili sperimentali.}
Il confronto tra zero-shot, one-shot e quattro scenari few-shot a
temperatura 0,2 varia l'informazione dimostrativa. Le configurazioni
few-shot a 0, 0,2 e 0,4 variano invece la temperatura degli evaluator.
Mantenere comuni modelli, istruzioni e criteri di aggiornamento rende
leggibili questi confronti, ma non elimina la variabilit\`a generativa:
temperatura zero non implica identit\`a garantita delle risposte.
Gli esempi fittizi non vengono ricavati dalle risposte della ground truth;
quest'ultima rimane il riferimento esterno per misurare il risultato.
Le configurazioni pianificate e i risultati storici sono distinti nel
Capitolo~\ref{ch:experiments}.
"""

EXPLANATIONS = r"""\subsubsection{Spiegazione dei quattro scenari e dei casi limite}
\label{sec:evaluator-example-rationale}

Le formulazioni tra virgolette seguenti riprendono gli input del codice;
le spiegazioni chiariscono le risposte attese, senza aggiungere requisiti
alle brevi fonti fittizie. Le etichette sono input gi\`a fissati:
gli esempi insegnano a usarle insieme alla documentazione, non a
ricalcolarle. Le nove decisioni coprono tutte e quattro le azioni,
con particolare attenzione ai confini tra conservazione, correzione
e perdita di funzionalit\`a.

\paragraph{Scenario 1: conservare un padre gi\`a completo.}
La fonte consente al richiedente di creare, modificare e cancellare
le proprie richieste. Il padre \emph{``Manage my service requests''}
ha descrizione \emph{``Create, modify and cancel my service requests.''};
HLG$'$ \`e \emph{``Submit and cancel my service requests''}, con
etichetta \texttt{EQUIVALENT} e nessun candidato duplicato.
La risposta \`e \texttt{KEEP} per il ramo \texttt{a}.
La modifica, assente dalla sintesi ricostruita, \`e gi\`a esplicitata
nel padre e supportata dalla fonte: chiederne l'aggiunta produrrebbe
una riscrittura priva di un difetto concreto. Il caso guida a leggere
anche la descrizione e a non assumere HLG$'$ come rappresentazione
completa e automaticamente migliore. Non autorizza \texttt{KEEP}
con \texttt{NOT\_EQUIVALENT}. Essendo il primo scenario, \`e anche
l'unica dimostrazione della condizione one-shot.

\paragraph{Scenario 2: tre difetti documentati che richiedono riscrittura.}
La fonte permette creazione, modifica e cancellazione delle proprie
richieste, riserva l'approvazione al funzionario e consente al richiedente
di consultarne l'esito. I tre rami ricevono \texttt{REWRITE}, per
ragioni differenti.
Nel ramo \texttt{a}, \emph{``Create service requests only''} esclude
esplicitamente modifiche e cancellazioni, mentre HLG$'$ \`e
\emph{``Create, modify and cancel my service requests''}
(\texttt{NOT\_EQUIVALENT}). Qui la fonte conferma una vera omissione
del padre: il feedback conserva la creazione e aggiunge le due
operazioni escluse, limitandole alle richieste dell'attore.
Nel ramo \texttt{b}, \emph{``Approve service requests''} assegna
approvazione e rifiuto al richiedente; HLG$'$ \`e
\emph{``Approve or reject service requests''} (\texttt{EQUIVALENT}).
L'equivalenza tra testi non rende corretta la responsabilit\`a:
il feedback richiede di attribuirla al funzionario e di distinguere
la consultazione dell'esito da parte del richiedente.
Nel ramo \texttt{c}, \emph{``Manage other requesters' service requests''}
permette di operare per qualunque richiedente, ma HLG$'$ \`e
\emph{``Manage my own service requests''} (\texttt{NOT\_EQUIVALENT}).
Il feedback restringe l'ambito alle proprie richieste, preservando
tutte e tre le operazioni. Il caso mostra che si possono correggere
copertura, responsabilit\`a o ambito senza eliminare una funzione valida;
il feedback descrive l'intervento richiesto e non garantisce che la
successiva generazione lo realizzi correttamente.

\paragraph{Scenario 3: distinguere assenza totale e supporto parziale.}
La fonte richiede prenotazioni di strumenti ed esclude espressamente
aste e vendite. Il ramo \texttt{a}, \emph{``Auction tools''}, descrive
l'organizzazione di aste; HLG$'$ \`e \emph{``Organize tool auctions''}
con \texttt{EQUIVALENT}. La risposta \`e \texttt{REMOVE}: la funzione
\`e interamente esclusa e non contiene una prenotazione da conservare.
Il ramo \texttt{b}, \emph{``Reserve and sell tools''}, ha HLG$'$
\emph{``Manage tool reservations and sales''}, ancora
\texttt{EQUIVALENT}, ma riceve \texttt{REWRITE}.
Il feedback conserva le prenotazioni e rimuove le vendite.
Eliminare il secondo ramo cancellerebbe anche una funzionalit\`a
richiesta. A parit\`a di etichetta, \`e dunque la verifica della fonte
e del contenuto del padre a distinguere le due azioni.

\paragraph{Scenario 4: ridondanza completa e sovrapposizione parziale.}
La fonte richiede creazione, modifica e cancellazione delle proprie
richieste, oltre al download delle ricevute. Tutti i rami hanno relazione
\texttt{EQUIVALENT} col proprio HLG$'$ e candidati
\texttt{POTENTIAL\_DUPLICATE}. Il ramo \texttt{a},
\emph{``Manage my service requests''}, comprende creazione, modifica
e cancellazione; HLG$'$ ripete queste tre operazioni.
Il ramo \texttt{b}, \emph{``Create my service requests''}, comprende
solo l'invio; HLG$'$ \`e \emph{``Submit a service request for myself''}.
Il ramo \texttt{c}, \emph{``Cancel requests and download receipts''},
comprende cancellazione e ricevute; HLG$'$ \`e
\emph{``Cancel my service requests and retrieve their receipts''}.
La risposta conserva \texttt{a}, elimina \texttt{b} con
\texttt{DUPLICATE\_REMOVAL} e \texttt{surviving\_branch\_id=a},
e conserva \texttt{c}. Tutte le funzioni di \texttt{b} sono gi\`a
coperte dal padre valido \texttt{a}; le ricevute di \texttt{c}, invece,
non lo sono. Il caso chiarisce perch\'e la candidatura non autorizzi
automaticamente la rimozione: occorre copertura completa del ramo
eliminato, senza trasferimenti di figli e senza perdere un risultato
supportato. Il superstite riceve \texttt{KEEP} nella stessa risposta.

Questi scenari sono scelti per rendere espliciti errori decisionali
plausibili: riscrittura cosmetica, fiducia nella sola equivalenza,
rimozione di un ramo parzialmente valido e confusione tra sovrapposizione
e duplicazione. Tale copertura motiva il loro uso nel prompt, ma non
ne dimostra l'ottimalit\`a. Poich\'e one-shot contiene soltanto lo
scenario \texttt{KEEP}, il confronto con few-shot varia anche la
copertura delle azioni dimostrate, non soltanto la lunghezza del prompt.
Gli esempi correnti documentano il codice corrente; non vengono
retroattivamente attribuiti a esecuzioni storiche con prompt diversi.
"""


def insert_block(text, key, anchor, body):
    start, end = f"% BEGIN {key}", f"% END {key}"
    block = f"{start}\n{body.rstrip()}\n{end}\n\n"
    if start in text:
        left = text.index(start)
        right = text.index(end, left) + len(end)
        return text[:left] + block.rstrip() + text[right:]
    assert text.count(anchor) == 1, anchor
    return text.replace(anchor, block + anchor)


def main():
    evaluator_path = ROOT / "src/bottom_up/global_goal_evaluator.py"
    source = evaluator_path.read_text(encoding="utf-8")
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "FEW_SHOT_EXAMPLES"
                              for t in n.targets))
    examples = ast.literal_eval(assignment.value)
    assert len(examples) == 4
    assert [len(e["response"]["decisions"]) for e in examples] == [1, 3, 2, 3]
    main_path = ROOT / "main.tex"
    text = main_path.read_text(encoding="utf-8")
    # Verify the existing complete listing is the actual current set, not a paraphrase.
    label = "label={lst:evaluator-examples-code}]"
    listing = text.split(label, 1)[1].split(r"\end{lstlisting}", 1)[0].strip()
    displayed = ast.literal_eval(ast.parse(listing).body[0].value)
    assert displayed == examples, "Evaluator listing differs from current implementation"
    text = insert_block(text, "DESIGN_RATIONALE", r"\section{Ingresso della revisione bottom-up}", RATIONALE)
    text = insert_block(text, "EXAMPLE_RATIONALE", r"\paragraph{Codice delle dimostrazioni e della selezione.}", EXPLANATIONS)
    main_path.write_text(text, encoding="utf-8")
    sources = [evaluator_path, ROOT / "src/llm_clients.py",
               ROOT / "src/bottom_up/goal_similarity.py", ROOT / "src/bottom_up/goal_orchestrator.py"]
    evidence = {
        "scope": "Current-code rationale and explanation of existing four fictional demonstrations; no new experimental results",
        "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "example_listing_matches_current_code": True,
        "scenario_decision_counts": [1, 3, 2, 3],
        "sections": {
            "sec:design-rationale": [str(p.relative_to(ROOT)) for p in sources],
            "sec:evaluator-example-rationale": [str(evaluator_path.relative_to(ROOT))],
        },
        "human_verification": {"status": "pending_author_review", "verifier": None, "date": None},
    }
    (OUT / "design_rationale_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print("Updated main.tex; verified 4 scenarios / 9 decisions against current evaluator listing.")


if __name__ == "__main__":
    main()
