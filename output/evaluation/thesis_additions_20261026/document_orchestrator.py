"""Document the current orchestrator and extract actual implementation listings."""
import ast
import hashlib
import inspect
import json
import re
import textwrap
from pathlib import Path

from src import llm_clients
from src.bottom_up import goal_orchestrator as orchestrator

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

section = r'''\section{Orchestratore e coordinamento delle chiamate LLM}
\label{sec:goal-orchestrator}
% Evidence O-01/O-02: goal_orchestrator.py, global_goal_evaluator.py, llm_clients.py.
Il coordinamento del metodo è affidato a un \emph{orchestratore software},
implementato in \path{src/bottom_up/goal_orchestrator.py}. Non si tratta di un
ulteriore LLM: è codice Python che stabilisce l'ordine delle operazioni,
trasferisce dati tra le funzioni, valida gli aggiornamenti e registra lo stato.
Le proposte degli LLM sono dati in ingresso a questo controllo; non modificano
direttamente la gerarchia e non scelgono autonomamente quali funzioni eseguire.

La funzione di ingresso adotta il nome \texttt{run\_global\_goal\_orchestrator},
per rendere esplicito il ruolo di coordinamento. Il precedente nome
\path{run_global_goal_cycle} rimane un alias della stessa funzione.
Il precedente modulo
\path{goal_cycle.py} è conservato come alias: entrambi i percorsi di import
risolvono allo stesso oggetto modulo, evitando copie dello stato o divergenze
nelle funzioni sostituite durante i test. I notebook correnti utilizzano il
nome \texttt{goal\_orchestrator}; i risultati storici mantengono la propria
provenienza e non vengono riscritti dalla rinomina.

\paragraph{Utilit\`a dell'orchestratore.}
Il vantaggio del coordinamento centrale consiste nel rendere esplicito chi
deve essere chiamato, con quali dati e in quale momento. I notebook avviano
una sola funzione; questa conserva lo snapshot comune, attende le risposte
valide prima dell'aggiornamento e inoltra al generatore soltanto le modifiche
approvate. Si evita cos\`i che il flusso sia distribuito tra celle o dipenda
dall'ordine con cui vengono ispezionate le risposte. L'orchestratore decide
la sequenza delle funzioni; il contenuto dei goal e i giudizi documentali
sono invece prodotti dai modelli e sottoposti ai controlli previsti.

\begin{samepage}
\paragraph{Instradamento delle chiamate ai modelli.}
Il percorso operativo distingue tre responsabilit\`a:
\begin{enumerate}
\item \textbf{Ricostruire e generare: Qwen.} La ricostruzione dei rami passa
da \texttt{reconstruct\_all\_branches} al client
\texttt{generate\_response}; lo stesso client viene usato dalle funzioni
di estrazione per generare gli HLG e gli LLG delle riparazioni.
\item \textbf{Giudicare lo snapshot: Ministral, Granite e Gemma.}
\texttt{evaluate\_all\_branches} riceve lo stato e interroga, uno alla
volta, i tre modelli configurati tramite \texttt{generate\_evaluator\_response}.
Ogni risposta globale viene validata; il voto viene aggregato per ramo
soltanto dopo aver raccolto le risposte utilizzabili. I modelli non si
passano messaggi e non leggono i giudizi degli altri.
\item \textbf{Controllare la riparazione: critic top-down.}
Quando il quorum approva una riscrittura, l'orchestratore invoca la
reflection: Qwen genera il candidato e il critic configurato, Ministral
per impostazione predefinita, ne restituisce punteggio e critica.
La reflection gestisce i propri tentativi prima di restituire il controllo
all'orchestratore, che applica o rifiuta l'aggiornamento secondo i controlli
HLG--LLG descritti di seguito.
\end{enumerate}
\end{samepage}
Le scelte dei modelli provengono dalla configurazione dei client, non
da una selezione autonoma compiuta da un LLM. Il calcolo MPNet della
similarit\`a \`e una fase numerica distinta da queste chiamate generative.
Il gestore locale rilascia il modello precedente prima di passare a un
modello diverso; la sequenza, i retry e i checkpoint sono approfonditi
nelle sottosezioni successive.

\subsection{Stato condiviso e ordine di esecuzione}
L'orchestratore riceve descrizione, attori, collezioni HLG e LLG iniziali e
configurazione della revisione. Ricostruisce i modelli Pydantic con le classi
correnti e lavora su copie profonde delle collezioni, preservando la baseline.
Riconcilia i riferimenti iniziali ai padri, unisce gli attori disponibili e
prepara il registro di configurazione, iterazioni e avvisi.

Ogni iterazione segue una sequenza esplicita:
\begin{enumerate}
\item \textbf{Snapshot e raggruppamento.} Indicizza gli HLG correnti e raccoglie
gli LLG associati. Gli identificativi di ramo valgono per lo snapshot corrente.
\item \textbf{Ricostruzione.} Invoca \texttt{reconstruct\_all\_branches} sui
soli gruppi non vuoti. La funzione usa Qwen e produce HLG$'$ senza consultare
il padre originale o la documentazione.
\item \textbf{Analisi numerica.} Invoca \texttt{analyze\_similarities}, che
usa MPNet e cosine, senza un LLM valutatore. Costruisce i dati diagnostici;
il modulo evaluator li trasforma nell'ingresso con etichette e coppie candidate.
\item \textbf{Revisione globale.} Invoca \texttt{evaluate\_all\_branches} una
volta per lo snapshot. Questa funzione interroga sequenzialmente i tre evaluator,
valida le risposte e aggrega i voti per ramo.
\item \textbf{Aggiornamento.} Invoca \texttt{apply\_review\_decisions}: applica
prima le rimozioni approvate e poi, se necessario, una riparazione top-down
dell'intera collezione superstite.
\item \textbf{Persistenza e arresto.} Salva la traccia completa e verifica
conferma dei rami e modifiche effettive; in alternativa avvia la nuova iterazione.
\end{enumerate}
La Figura~\ref{fig:orchestrator-call-flow} mostra questo flusso. Gli evaluator
esaminano lo stesso stato precedente agli aggiornamenti; non votano su gerarchie
modificate dalle decisioni dei modelli interrogati prima di loro.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
 node distance=0.45cm,
 phase/.style={draw,rounded corners,align=center,text width=0.83\linewidth,
               inner sep=5pt,font=\small},
 connection/.style={-{Latex},thick}]
\node[phase] (state) {\texttt{run\_global\_goal\_orchestrator}: snapshot HLG--LLG e gruppi};
\node[phase,below=of state] (prime) {\texttt{reconstruct\_all\_branches}: Qwen produce HLG$'$};
\node[phase,below=of prime] (similarity) {\texttt{analyze\_similarities}: MPNet, cosine, candidati};
\node[phase,below=of similarity] (review) {\texttt{evaluate\_all\_branches}: tre evaluator in sequenza, risposte globali, validazione e maggioranza};
\node[phase,below=of review] (update) {\texttt{apply\_review\_decisions}: rimozioni e riparazione top-down con generatore e critic};
\node[phase,below=of update] (save) {Checkpoint completo, controllo di convergenza o nuova iterazione};
\draw[connection] (state)--(prime);
\draw[connection] (prime)--(similarity);
\draw[connection] (similarity)--(review);
\draw[connection] (review)--(update);
\draw[connection] (update)--(save);
\end{tikzpicture}
\caption{Coordinamento deterministico di una iterazione. Le frecce rappresentano
passaggi di dati e chiamate Python, non una conversazione tra agenti.}
\label{fig:orchestrator-call-flow}
\end{figure}

\subsection{Funzioni passate come argomento e callback}
La funzione locale \texttt{run\_stage(name, function, *args, **kwargs)} riceve
una funzione Python e i suoi argomenti, quindi esegue
\texttt{function(*args, **kwargs)}. Il parametro \texttt{name} identifica la
fase nel punto di chiamata, ma non è usato per scegliere un modello o una
strategia. Con \texttt{verbose=True} gli output della funzione sono visibili;
altrimenti stdout e stderr vengono catturati durante l'esecuzione. Il wrapper
non aggiunge retry, non cambia il prompt e non applica una decisione semantica.

Anche la riparazione è passata come funzione: l'argomento \texttt{repair}
di \texttt{apply\_review\_decisions} è una funzione anonima che richiama
\texttt{\_repair\_collections\_with\_top\_down} attraverso
\texttt{run\_stage}. Questo separa l'applicazione delle decisioni dalla
procedura di generazione e consente ai test di sostituire la riparazione
senza contattare i modelli.

Il controllo dei salvataggi usa callback esplicite. Dopo la risposta di un
evaluator, \texttt{save\_global\_vote(record)} aggiunge il record e richiama
\texttt{checkpoint}. Quest'ultima costruisce una traccia marcata
\texttt{checkpoint\_complete=False}, che descrive un'iterazione ancora in
corso. Il callback \texttt{on\_evaluation=checkpoint} riceve poi le decisioni
aggregate. La funzione \texttt{publish} serializza lo stato, scrive il JSON
se è configurata una directory e chiama l'eventuale \texttt{on\_state}
fornito dal notebook o dal gestore della suite. Solo dopo l'aggiornamento
della gerarchia la traccia è completata e aggiunta alle iterazioni concluse.
\texttt{finish} costruisce il risultato finale e lo pubblica con il motivo
di arresto. Un checkpoint parziale non viene contato come iterazione conclusa.

\subsection{Dalla funzione del compito alla richiesta al modello}
Le funzioni del compito costruiscono gli input; il trasporto delle richieste
è centralizzato in \path{src/llm_clients.py}.
\begin{description}
\item[Generatore e ricostruttore.]
\texttt{reconstruct\_high\_level\_goal},
\texttt{generate\_high\_level\_goals} e
\texttt{generate\_low\_level\_goals} chiamano \texttt{generate\_response}.
Questa funzione prepara messaggi di sistema e utente, richiede a Qwen un
risultato JSON conforme al modello Pydantic, controlla troncamenti e output
vuoti e valida la risposta. La temperatura del generatore è fissa a zero;
le temperature sperimentali riguardano gli evaluator globali.
Le ricostruzioni richiedono retry senza includere le risposte scartate;
altri chiamanti possono usare la correzione JSON prevista dal client.
\item[Evaluator bottom-up.]
\texttt{evaluate\_all\_branches} chiama \texttt{\_critic1\_vote} per ogni
modello configurato. Quest'ultima passa prompt, modello, temperatura e schema
a \texttt{generate\_evaluator\_response}. Il client pubblico gestisce il
modello locale e delega la richiesta a \texttt{\_generate\_evaluator\_response},
che usa \path{/v1/chat/completions} con \texttt{response\_format=json\_schema}.
\item[Critic della riparazione top-down.]
\texttt{generate\_response\_with\_reflection} riceve il generatore come
\texttt{call\_function}, i suoi argomenti in \texttt{define\_args} e quelli
della valutazione in \texttt{eval\_args}. Genera il candidato, chiama
\texttt{get\_evaluation} sul critic configurato, legge punteggio e critica
e, se necessario, ripete con un oggetto \texttt{Feedback}. Questa valutazione
testuale usa lo stesso client degli evaluator, ma non il loro schema JSON
o il voto di maggioranza. L'adattatore bottom-up inietta la critica iniziale
soltanto nella prima chiamata; i retry successivi appartengono alla reflection.
\end{description}

\subsection{Modelli locali, indipendenza dei giudizi e limiti dei retry}
Il gestore \texttt{loaded\_local\_model} delimita l'uso di un modello mediante
un context manager. Un lock rientrante e il riferimento al modello attivo
coordinano le chiamate nel processo Python: invocazioni annidate dello stesso
modello riusano l'istanza, mentre una richiesta per un modello diverso durante
un uso ancora attivo genera un errore. Il gestore interno verifica l'istanza
e il contesto disponibile, carica il modello quando necessario e lo rilascia
all'uscita anche in caso di errore. Gli altri modelli della pipeline vengono
scaricati prima del caricamento selezionato. Il lock è locale al processo:
non sincronizza automaticamente notebook o programmi Python differenti.
Durante la ricostruzione, \texttt{reconstruct\_all\_branches} mantiene aperto
un unico uso esterno di Qwen per tutti i rami: le richieste interne riusano
quel modello. Nella reflection, invece, generatore e critic si alternano,
rilasciando il modello precedente prima di richiamare quello successivo.

I tre evaluator sono quindi interrogati uno alla volta, senza scambio di
messaggi o accesso ai voti altrui. La separazione riguarda le chiamate e i
giudizi prodotti sullo stesso input, non un'esecuzione parallela né una
garanzia statistica di errori indipendenti.
Una risposta globale con JSON non valido, ID mancanti, ID duplicati o azioni
non ammissibili viene scartata interamente; non si combinano pezzi di tentativi
diversi dello stesso modello. Il quorum richiede due voti validi concordi
sull'azione e, per la duplicazione, sul medesimo superstite. Soltanto le
critiche dei voti concordanti vengono consolidate per il generatore.

L'orchestratore distingue limiti diversi: iterazioni globali, tentativi
tecnici della similarità, tentativi della risposta globale di ciascun evaluator
e tentativi della reflection top-down. Per errori di contesto l'evaluator
interessato può essere ricaricato con una richiesta di contesto raddoppiata,
entro il suo stesso budget di tentativi. Non viene troncato lo snapshot.
La riparazione aggiorna insieme HLG e LLG soltanto dopo l'approvazione degli
HLG e la validazione strutturale dei figli. Se fallisce, conserva lo stato
precedente alla riparazione, incluse le rimozioni già approvate. Le condizioni
di aggiornamento e arresto sono dettagliate nella Sezione~\ref{sec:approach-loop}.

Gli estratti reali delle funzioni sono riportati nella
Sezione~\ref{sec:orchestrator-implementation}; il codice è distinto dai prompt
che i modelli ricevono e non viene interpretato come istruzione di un agente.

'''

# Permit line breaks in long Python identifiers without changing their spelling.
section = re.sub(r'\\texttt\{([^{}]+)\}',
                 lambda m: '\\path{' + m.group(1).replace(r'\_', '_') + '}'
                 if r'\_' in m.group(1) else m.group(0), section)


def code_listing(caption, label, source):
    return ('\\begin{lstlisting}[style=llmprompt,language=Python,caption={' + caption
            + '},label={' + label + '}]\n' + source.rstrip() + '\n\\end{lstlisting}\n\n')


def extract_node(source, name):
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    segment = ast.get_source_segment(source, node)
    if node.col_offset:
        lines = segment.splitlines()
        segment = '\n'.join([lines[0]] + [line[node.col_offset:] for line in lines[1:]])
    return segment


cycle_source = inspect.getsource(orchestrator.run_global_goal_orchestrator)
section += r'''\subsection{Il coordinamento nelle funzioni Python}
\label{sec:orchestrator-code-walkthrough}
Il Listing~\ref{lst:approach-orchestrator-stage} mostra il wrapper effettivo
definito nella funzione principale. Riceve una funzione e i suoi argomenti:
la esegue direttamente se i log sono visibili, altrimenti ne cattura stdout
e stderr. Non sceglie il modello e non modifica il prompt. Questa distinzione
separa il controllo del flusso dall'implementazione dei singoli compiti.

'''
section += code_listing('Funzione locale di esecuzione delle fasi, estratta dal codice corrente.',
                        'lst:approach-orchestrator-stage', extract_node(cycle_source, 'run_stage'))
source_lines = cycle_source.splitlines()
begin = next(i for i, line in enumerate(source_lines) if line.strip() == 'update = apply_review_decisions(')
end = next(i for i, line in enumerate(source_lines[begin:], begin)
           if line.strip() == 'changed, confirmed = update["changed"], update["confirmed"]')
update_source = textwrap.dedent('\n'.join(source_lines[begin:end+1]))
section += r'''Dopo l'aggregazione dei voti, il Listing~\ref{lst:approach-orchestrator-update}
mostra il punto di applicazione delle decisioni. L'orchestratore passa le
collezioni correnti e una funzione di riparazione, poi riceve nuove
collezioni, esiti, indicatore di modifica e rami confermati. La funzione
anonima non avvia un nuovo agente: richiama la riparazione top-down con
gli argomenti stabiliti dal codice. Le variabili dell'estratto sono
preparate nelle fasi precedenti della stessa iterazione; il frammento
non costituisce uno script autonomo.

'''
section += '\\noindent\\begin{minipage}{\\linewidth}\n' + code_listing('Estratto contiguo del passaggio dalle decisioni aggregate allo stato aggiornato.',
                        'lst:approach-orchestrator-update', update_source) + '\\end{minipage}\n\n'
section += r'''Le definizioni integrali di
\path{run_global_goal_orchestrator} e \path{apply_review_decisions}
sono riportate nei Listing~\ref{lst:orchestrator-main-full}
e~\ref{lst:orchestrator-apply-full}. La prima funzione controlla
configurazione, sequenza, checkpoint e arresto; la seconda applica prima
le rimozioni, raccoglie i feedback di riscrittura e tenta un aggiornamento
batch. Se questo fallisce, conserva lo stato precedente alla riparazione,
incluse le rimozioni gi\`a applicate. Tale separazione rende verificabile
il motivo per cui una proposta di un LLM non coincide necessariamente
con una modifica della gerarchia.

'''
appendix = r'''\section{Funzioni dell'orchestratore e delle chiamate ai modelli}
\label{sec:orchestrator-implementation}
Questa sezione riproduce estratti del codice corrente. Le funzioni locali sono
definite dentro \texttt{run\_global\_goal\_orchestrator} e usano il suo stato tramite
closure: non sono procedure autonome da eseguire separatamente.
La funzione principale mantiene il controllo della sequenza descritta nella
Sezione~\ref{sec:goal-orchestrator}. Non sono richieste conversazioni tra LLM
per decidere quale stadio invocare.

'''
appendix += '\\clearpage\n' + code_listing('Definizione integrale del punto di ingresso dell\'orchestratore: configurazione, fasi, aggiornamento e arresto.',
                         'lst:orchestrator-main-full', cycle_source)
appendix += code_listing('Applicazione delle decisioni: rimozioni approvate, raccolta delle critiche e riparazione batch.',
                         'lst:orchestrator-apply-full', inspect.getsource(orchestrator.apply_review_decisions))
from src.bottom_up.global_goal_evaluator import _validate_decision
appendix += code_listing('Validatore operativo corrente delle singole decisioni: i controlli strutturali sono distinti dalle condizioni documentali del prompt.',
                         'lst:documentary-decision-validator', inspect.getsource(_validate_decision))
appendix += '\\clearpage\n' + code_listing('Wrapper che esegue una funzione di fase e controlla soltanto la visibilità dei log.',
                         'lst:orchestrator-run-stage', extract_node(cycle_source, 'run_stage'))
callbacks = '\n\n'.join(extract_node(cycle_source, name) for name in
                         ['save_global_vote', 'checkpoint', 'publish', 'finish'])
appendix += code_listing('Callback e salvataggio dello stato: i checkpoint parziali restano distinti dalle iterazioni complete.',
                         'lst:orchestrator-state-callbacks', callbacks)
review_source = inspect.getsource(orchestrator.evaluate_all_branches)
model_loop = next(node for node in ast.parse(review_source).body[0].body
                  if isinstance(node, ast.For) and isinstance(node.iter, ast.Name)
                  and node.iter.id == 'BOTTOM_UP_EVALUATOR_MODELS')
loop_lines = ast.get_source_segment(review_source, model_loop).splitlines()
loop_code = '\n'.join([loop_lines[0]] + [line[model_loop.col_offset:] for line in loop_lines[1:]])
appendix += code_listing('Ciclo reale di interrogazione dei tre evaluator: ogni risposta globale viene salvata prima di passare al modello successivo.',
                         'lst:orchestrator-evaluator-sequence', loop_code)
appendix += code_listing('Adattamento della chiamata alla reflection e registrazione dei candidati della riparazione.',
                         'lst:orchestrator-top-down-call', inspect.getsource(orchestrator._run_original_top_down))
appendix += code_listing('Gestore rientrante dei modelli locali: stesso modello nelle chiamate annidate e rilascio al termine.',
                         'lst:orchestrator-model-lease', inspect.getsource(llm_clients.loaded_local_model))
appendix += code_listing('Client del generatore Qwen: messaggi, schema richiesto e validazione del risultato.',
                         'lst:orchestrator-generator-client', inspect.getsource(llm_clients.generate_response))
appendix += code_listing('Client degli evaluator: gestione del modello e invio della richiesta testuale o con schema JSON.',
                         'lst:orchestrator-evaluator-client',
                         inspect.getsource(llm_clients.generate_evaluator_response) + '\n\n'
                         + inspect.getsource(llm_clients._generate_evaluator_response))
appendix += r'''La richiesta strutturata viene ulteriormente verificata da
\texttt{\_critic1\_vote} prima di partecipare al voto. La gestione del retry
globale e l'applicazione degli aggiornamenti rimangono separate dal client HTTP.
L'adattatore che inserisce il feedback soltanto al primo tentativo e la funzione
di riparazione della collezione sono riportati anche nei
Listing~\ref{lst:bottom-up-feedback-adapter} e~\ref{lst:collection-repair-code}.

'''

from output.evaluation.thesis_additions_20261026.bottom_up_code_annotations import enrich
section, appendix = enrich(section, appendix, code_listing)

p = ROOT / 'main.tex'
text = p.read_text(encoding='utf-8')
start_marker = r'\section{Orchestratore e coordinamento delle chiamate LLM}'
next_marker = r'\section{Aggiornamento della gerarchia e iterazioni}'
if start_marker in text:
    start = text.index(start_marker)
    end = text.index(next_marker, start)
    text = text[:start] + section + text[end:]
else:
    text = text.replace(next_marker, section + next_marker, 1)
start_marker = r"\section{Funzioni dell'orchestratore e delle chiamate ai modelli}"
next_marker = r'\section{Prompt della riparazione top-down}'
if start_marker in text:
    start = text.index(start_marker)
    end = text.index(next_marker, start)
    text = text[:start] + appendix + text[end:]
else:
    text = text.replace(next_marker, appendix + next_marker, 1)
text = text.replace(r'\path{src/bottom_up/goal_cycle.py} contiene ciclo e aggiornamenti;',
                    r'\path{src/bottom_up/goal_orchestrator.py} contiene orchestrazione e aggiornamenti;')
text = text.replace('Una funzione principale collega gli output delle fasi senza introdurre\nun agente decisionale aggiuntivo.',
                    "L'orchestratore descritto nella Sezione~\\ref{sec:goal-orchestrator}\ncollega gli output delle fasi senza introdurre un agente decisionale aggiuntivo.")
p.write_text(text, encoding='utf-8')
(OUT / 'orchestrator_section.tex').write_text(section, encoding='utf-8')
(OUT / 'orchestrator_implementation.tex').write_text(appendix, encoding='utf-8')
sources = [
    'src/bottom_up/goal_orchestrator.py', 'src/bottom_up/goal_cycle.py',
    'src/bottom_up/goal_reconstructor.py', 'src/bottom_up/goal_similarity.py',
    'src/bottom_up/global_goal_evaluator.py', 'src/llm_clients.py',
    'src/extraction/extractor.py', 'src/self_critique/refine_response.py',
]
evidence = {'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
            'claims': {'O-01': 'Orchestration, stage callables, callbacks, removal/repair transaction and stop conditions',
                       'O-02': 'Sequential evaluator calls, whole-response validation, majority and local process model leases'},
            'listings': 'Extracted with inspect/AST; thesis-only numbered comments; executable AST equivalence and compilation checked; nested functions retain their closure dependencies',
            'human_author_verification': 'pending'}
(OUT / 'orchestrator_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('Added orchestrator explanation, call flow diagram, exact function listings and source hashes.')
