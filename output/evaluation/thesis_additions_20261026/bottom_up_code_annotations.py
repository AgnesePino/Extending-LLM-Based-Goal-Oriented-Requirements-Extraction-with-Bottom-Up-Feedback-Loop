"""Add thesis-only comments while proving the displayed Python AST unchanged."""
import ast
import inspect
import re

from src.bottom_up import global_goal_evaluator as review
from src.bottom_up import goal_reconstructor as recon
from src.bottom_up import goal_similarity as similarity


def annotate(source, notes):
    """Insert comments before matched statements; never edit an executable line."""
    pending = list(notes)
    lines = []
    number = 0
    for line in source.splitlines():
        for prefix, comment in list(pending):
            if line.lstrip().startswith(prefix):
                number += 1
                indent = line[:len(line)-len(line.lstrip())]
                lines.append(f'{indent}# [{number:02d}] {comment}')
                pending.remove((prefix, comment))
        lines.append(line)
    assert not pending, f'Missing annotation targets: {pending}'
    result = '\n'.join(lines)
    assert ast.dump(ast.parse(source), include_attributes=False) == ast.dump(ast.parse(result), include_attributes=False)
    compile(result, '<annotated thesis listing>', 'exec')
    return result


def annotate_listing(text, label, notes):
    pattern = r'(\\begin\{lstlisting\}\[[^\n]*label=\{' + re.escape(label) + r'\}\]\n)(.*?)(\n\\end\{lstlisting\})'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    assert len(matches) == 1, label
    m = matches[0]
    return text[:m.start(2)] + annotate(m.group(2), notes) + text[m.end(2):]


def enrich(section, appendix, listing):
    section += r'''\subsection{Mappa delle funzioni e lettura del codice annotato}
\label{sec:bottom-up-annotated-map}
I commenti numerati \texttt{\# [01]}, \texttt{\# [02]} e successivi nei
listing sono annotazioni esplicative aggiunte nella tesi. Le istruzioni
eseguibili rimangono quelle del codice corrente; il confronto degli alberi
sintattici Python verifica che siano cambiati soltanto i commenti.
La numerazione riparte per ogni funzione. La tabella seguente collega
le chiamate dell'orchestratore ai blocchi commentati dell'appendice.

\begin{table}[htbp]
\centering\small
\caption{Funzioni delle fasi del ciclo bottom-up e punti da leggere nel codice annotato.}
\label{tab:bottom-up-annotated-functions}
\begin{tabularx}{\textwidth}{@{}p{0.48\textwidth}X@{}}
\toprule
\textbf{Funzione} & \textbf{Responsabilit\`a e listing} \\
\midrule
\path{run_global_goal_orchestrator} & Coordina stato, fasi e arresto; Listing~\ref{lst:orchestrator-main-full}. \\
\path{reconstruct_all_branches} & Invoca Qwen sui gruppi di figli; Listing~\ref{lst:annotated-reconstruction}. \\
\path{_encode_text}, \path{_cosine} & Producono e confrontano vettori validi; Listing~\ref{lst:annotated-numerical-primitives}. \\
\path{_analyze_similarities_once}, \path{analyze_similarities} & Costruiscono etichette e candidati, gestendo gli errori tecnici; Listing~\ref{lst:annotated-similarity-pass} e~\ref{lst:annotated-similarity-retry}. \\
\path{build_review_payload} & Prepara l'ingresso testuale degli evaluator; Listing~\ref{lst:annotated-review-payload}. \\
\path{_critic1_vote} & Ottiene una risposta globale valida per un modello; Listing~\ref{lst:annotated-model-vote}. \\
\path{evaluate_all_branches} & Interroga i tre modelli e approva il quorum per ramo; Listing~\ref{lst:annotated-global-voting}. \\
\path{_merge_agreeing_judgments} & Unisce soltanto le critiche concordi; Listing~\ref{lst:annotated-feedback-merge}. \\
\path{apply_review_decisions} & Applica rimozioni e riparazione batch; Listing~\ref{lst:orchestrator-apply-full}. \\
\bottomrule
\end{tabularx}
\end{table}

La catena va letta nell'ordine \emph{ricostruzione, analisi numerica,
preparazione dell'ingresso, risposte dei modelli, aggregazione e
aggiornamento}. Le chiamate ai client locali rimangono nei rispettivi
helper. Il nome storico \path{_critic1_vote} identifica la funzione
comune usata per ciascuno dei tre evaluator, non un voto riservato
soltanto al primo modello.

'''
    appendix = annotate_listing(appendix, 'lst:orchestrator-main-full', [
        ('if mode !=', 'Verifica le condizioni della pipeline top-down riusata.'),
        ('current_hlgs = _coerce_model', 'Copia la baseline: le revisioni lavorano sullo stato corrente.'),
        ('current_llgs = _reconcile_parent_references(', 'Riconcilia i riferimenti prima del raggruppamento dei figli.'),
        ('config = {', 'Registra policy e parametri per ricostruire questa esecuzione.'),
        ('def publish(', 'Pubblica uno stato finale oppure un checkpoint ancora parziale.'),
        ('def finish(', 'Costruisce il risultato con il motivo di arresto.'),
        ('for iteration in range(', 'Il limite globale resta fisso; aumenta solo il contatore.'),
        ('branches = indexed_high_level_goals(', 'Gli ID di ramo appartengono a questo snapshot.'),
        ('reconstructions = {', 'Qwen ricostruisce HLG prime soltanto per gruppi non vuoti.'),
        ('analysis = run_stage(', 'MPNet calcola relazioni e candidati senza un giudizio LLM.'),
        ('def save_global_vote(', 'Salva ogni risposta di modello prima degli aggiornamenti.'),
        ('evaluations = run_stage(', 'I tre evaluator ricevono lo stesso snapshot e contesto.'),
        ('update = apply_review_decisions(', 'Applica le sole proposte approvate e tenta la riparazione.'),
        ('trace = GlobalGoalCycleIteration(', 'Costruisce un checkpoint parziale con dati e voti correnti.'),
        ('traces.append(trace)', 'Archivia la traccia completa solo dopo gli aggiornamenti della gerarchia.'),
        ('if not changed and len(confirmed)', 'Arresta per conferma solo senza cambiamenti e con tutti i rami confermati.'),
        ('return finish(GlobalGoalCycleStopReason.MAX_', 'Conserva il risultato anche se il limite viene raggiunto senza convergenza.'),
    ])
    appendix = annotate_listing(appendix, 'lst:orchestrator-apply-full', [
        ('next_goals = ', 'Lavora sulle collezioni correnti indicizzate per ramo.'),
        ('protected = ', 'Protegge i rami indicati come superstiti da rimozioni incompatibili.'),
        ('next_goals.pop(branch)', 'Rimuove il padre; i figli associati sono filtrati subito dopo.'),
        ('for bid, result in evaluations.items():', 'Assegna un esito anche ai rami senza interventi strutturali.'),
        ('hlg_requests', 'Raccoglie le critiche dei rami superstiti da riscrivere.'),
        ('if requested_ids:', 'Una sola riparazione batch riceve tutti i feedback raccolti.'),
        ('state_changed = ', 'Confronta i contenuti: handoff riuscito non implica cambiamento.'),
        ('except Exception as exc:', 'Se la riparazione fallisce conserva lo stato dopo le rimozioni.'),
        ('confirmed = ', 'Dopo un handoff la nuova gerarchia richiede una nuova verifica.'),
    ])
    appendix += r'''\subsection{Fasi del bottom-up: definizioni con annotazioni}
\label{sec:annotated-bottom-up-phases}
Le funzioni seguenti sono estratte dai moduli correnti. I commenti
numerati spiegano input, chiamate e controlli nel punto in cui avvengono;
non sono istruzioni aggiunte ai prompt e non modificano il runtime.

\paragraph{Ricostruzione dei rami.}
Il caricamento esterno mantiene Qwen disponibile mentre si attraversano
i gruppi di figli. La funzione di ricostruzione del singolo gruppo
costruisce il prompt e il risultato riporta gli LLG usati come sorgente.
'''
    specs = [
        ('lst:annotated-reconstruction', 'Ricostruzione dei rami e provenienza degli HLG ricostruiti.', recon.reconstruct_all_branches, [
            ('if isinstance(goal_lists', 'Accetta anche le collezioni top-down e raggruppa i figli per padre.'),
            ('if not goal_lists:', 'Un insieme vuoto non richiede chiamate al modello.'),
            ('with loaded_local_model(', 'Mantiene lo stesso modello caricato per tutti i gruppi.'),
            ('output = reconstruct_high_level_goal(', 'La funzione del singolo ramo usa soltanto le descrizioni dei figli.'),
            ('results[branch_id] = ', 'Registra testo ricostruito e nomi dei figli sorgente.'),
        ]),
        ('lst:annotated-numerical-primitives', 'Embedding e cosine: controlli numerici prima delle etichette.', None, []),
        ('lst:annotated-similarity-pass', 'Confronti HLG prime contro tutti gli HLG originali del progetto.', similarity._analyze_similarities_once, [
            ('for goal_id, parent in branches.items():', 'Codifica le descrizioni originali e le ricostruzioni disponibili.'),
            ('for goal_id in ids:', 'Ogni riga parte da una ricostruzione; le colonne sono HLG originali.'),
            ('own = comparisons[', 'Il confronto diagonale identifica il rapporto con il proprio padre.'),
            ('candidate = (', 'Soglia, disponibilita e stesso attore delimitano i candidati duplicati.'),
            ('record["duplicate_label"] = ', 'La candidatura numerica resta da confermare sulla documentazione.'),
        ]),
        ('lst:annotated-similarity-retry', 'Recupero degli errori tecnici della similarita, con budget separato.', similarity.analyze_similarities, [
            ('for attempt in range(', 'Riprova soltanto entro il budget tecnico della similarita.'),
            ('if not errors:', 'Senza errori tecnici termina i retry; un ramo puo restare non ricostruibile.'),
            ('_embedding_model.cache_clear()', 'Ricarica il modello dopo errori; conserva i vettori validi in cache.'),
            ('analysis["technical_attempts"] = ', 'Conserva la storia dei tentativi anche dopo il recupero.'),
        ]),
        ('lst:annotated-review-payload', 'Ingresso degli evaluator: testi, etichette e coppie candidate.', similarity.build_review_payload, [
            ('return {', 'Separa gli ID richiesti dalle informazioni di contesto della gerarchia.'),
            ('"potential_duplicates":', 'Ogni coppia contiene HLG prime e il suo specifico HLG candidato.'),
        ]),
        ('lst:annotated-model-vote', 'Risposta globale di un evaluator: chiamata, validazione e retry.', review._critic1_vote, [
            ('eligible = ', 'Richiede decisioni soltanto per confronti numerici disponibili.'),
            ('for attempt in range(', 'Lo stesso budget copre caricamento, contesto e risposta.'),
            ('raw = generate_evaluator_response(', 'Invia modello, temperatura, prompt e schema al client locale.'),
            ('if set(counts) != ', 'Una risposta deve coprire esattamente gli ID richiesti una sola volta.'),
            ('vote = _validate_decision(', 'Controlla ciascuna azione prima di accettare la risposta intera.'),
            ('success = ', 'Nessun voto parziale di un tentativo scartato viene riutilizzato.'),
            ('if context_error:', 'Un errore di contesto puo richiedere il raddoppio del contesto.'),
            ('global_record = ', 'Restituisce voti accettati e cronologia del modello.'),
        ]),
        ('lst:annotated-global-voting', 'Giudizi dei tre modelli, quorum per ramo e verifica dei superstiti.', review.evaluate_all_branches, [
            ('prompt = _critic1_prompt(', 'Costruisce un solo prompt di snapshot condiviso dai tre modelli.'),
            ('for model in BOTTOM_UP_EVALUATOR_MODELS:', 'Interroga i modelli in sequenza senza mostrare i voti precedenti.'),
            ('counts = Counter(', 'La chiave conta azione e superstite, non la sola azione.'),
            ('if not counts or ', 'Senza due operazioni concordi il ramo resta inconcludente.'),
            ('vote = _merge_agreeing_judgments(', 'Unisce solo i feedback dei voti che sostengono la maggioranza.'),
            ('kept = {', 'Controlla i superstiti sullo stesso aggregato, senza dipendenze di ordine.'),
            ('if survivor and survivor not in kept:', 'Respinge una duplicazione se il superstite non e confermato KEEP.'),
        ]),
        ('lst:annotated-feedback-merge', 'Consolidamento del feedback dei soli voti concordi.', review._merge_agreeing_judgments, [
            ('votes = [', 'Scarta dal feedback i giudizi che non sostengono la stessa operazione.'),
            ('return "\\n\\n".join(', 'Elimina solo critiche testualmente identiche; non risolve contraddizioni.'),
        ]),
    ]
    for label, caption, fn, notes in specs:
        if fn is None:
            source = annotate(inspect.getsource(similarity._encode_text), [
                ('if not text.strip():', 'Un testo vuoto non riceve un vettore sostitutivo.'),
                ('if len(tokens) > ', 'Un testo troppo lungo genera un errore invece di essere troncato.'),
                ('vector = ', 'Codifica il testo con normalizzazione del vettore.'),
            ]) + '\n\n' + annotate(inspect.getsource(similarity._cosine), [
                ('return float(', 'Normalizza i vettori e limita il risultato in [-1, 1].'),
            ])
        else:
            source = annotate(inspect.getsource(fn), notes)
        appendix += listing(caption, label, source)
    return section, appendix
