"""Analyse completed v111 traces without inference or changes to run archives."""
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'output/evaluation/evaluator_dynamics_20261026'
OUT.mkdir(parents=True, exist_ok=True)
SUITE = ROOT / 'output/bottom_up_experiments/ofat_fixed_thresholds_20261025_170859'
CONFIGS = ['E1_0_examples', 'E1_1_examples', 'reference']
LABELS = dict(zip(CONFIGS, ['P0', 'P1', 'P4']))
MODELS = {'mistralai/ministral-3-14b-reasoning': 'Ministral',
          'ibm/granite-3.2-8b': 'Granite', 'google/gemma-4-12b-qat': 'Gemma'}
SHORT = {'Assegno Unico Universale - SIA Project 24 25': 'AUU',
 'Ethical Purchasing Group - SIA Project 22 23': 'Ethical Purchasing',
 'Event Organization Portal - SIA Project 21 22': 'Event Organization',
 'La Reine Marlene - SIA Project 23 24': 'La Reine Marlene',
 'London Ambulance Service': 'London Ambulance', 'SIA Project 25 26': 'SIA 25--26'}
ACTION = {'KEEP_ORIGINAL_HIGH_LEVEL_GOAL': 'K', 'REWRITE_ORIGINAL_HIGH_LEVEL_GOAL': 'W',
          'REMOVE_ORIGINAL_HIGH_LEVEL_GOAL': 'R', 'DUPLICATE_REMOVAL': 'D'}


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def esc(x):
    return ''.join({'&': r'\&', '_': r'\_', '%': r'\%', '#': r'\#', '$': r'\$'}.get(c,c) for c in str(x))


def num(x, digits=1):
    return f'{x:.{digits}f}'.replace('.', ',')


def table(caption, label, headings, rows, columns=None):
    return '\n'.join([r'\begin{table}[!htbp]', r'\centering\small',
        '\\caption{' + caption + '}\\label{' + label + '}',
        '\\begin{tabular}{' + (columns or 'l' + 'r'*(len(headings)-1)) + '}',
        r'\toprule', ' & '.join(headings) + r' \\', r'\midrule',
        *[' & '.join(map(str, row)) + r' \\' for row in rows],
        r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def figure(name, caption, label):
    return ('\\begin{figure}[htbp]\n\\centering\n'
            '\\includegraphics[width=\\textwidth]{output/evaluation/evaluator_dynamics_20261026/'
            + name + '.pdf}\n\\caption{' + caption + '}\\label{' + label + '}\n\\end{figure}\n')


def fingerprint(hlgs, llgs):
    return hashlib.sha256(json.dumps([hlgs, llgs], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def operation(d):
    return (d['decision'], d.get('surviving_branch_id') or '')


requests, attempts, votes, branches, iterations, runs, pair_votes = [], [], [], [], [], [], []
hashes = {}
documents = {}
snapshots = []
for path in sorted((SUITE / 'datasets').glob('*.json')):
    raw = path.read_bytes()
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    doc = json.loads(raw)
    name = doc['name']; documents[name] = doc
    for config in CONFIGS:
        run = doc['runs'][config+'/repeat_001']
        assert run['status'] == 'completed'
        traces = run['globalGoalCycle']['iterations']
        assert len(traces) == run['completed_iterations']
        identity = dict(dataset=name,config_id=config,repetition=1)
        seen = {fingerprint(doc['baseline']['structuredHighLevelGoals'], doc['baseline']['structuredLowLevelGoals'])}
        changed_count = 0; unchanged_count = 0; revisits = 0
        for t in traces:
            row = dict(identity, iteration=t['iteration'])
            before = fingerprint(t['input_hierarchy']['high_level_goals'], t['input_hierarchy']['low_level_goals'])
            after = fingerprint(t['high_level_goals'], t['low_level_goals'])
            changed = before != after
            changed_count += changed; unchanged_count += not changed
            revisit = changed and after in seen
            revisits += revisit; seen.add(after)
            iterations.append(dict(row, branches=len(t['evaluations']), changed=changed, revisited_state=revisit))
            by_model = {}
            for record in t['evaluator_responses']:
                model = record['model']; valid = record['status'] == 'VALID'
                requests.append(dict(row, model=model, valid=valid, attempts=len(record['attempts'])))
                for a in record['attempts']:
                    rejected = bool(a.get('errors'))
                    attempts.append(dict(row, model=model, attempt=a['attempt'], rejected=rejected,
                        error=json.dumps(a.get('errors',{}),ensure_ascii=False),
                        prompt_characters=a.get('generation',{}).get('prompt_characters')))
                by_model[model] = {d['branch_id']: d for d in record.get('decisions',[])} if valid else {}
                for d in record.get('decisions',[]) if valid else []:
                    votes.append(dict(row,model=model,branch_id=d['branch_id'],action=ACTION[d['decision']],
                        relationship=t['similarity_analysis']['branches'][d['branch_id']]['relationship'],
                        survivor=d.get('surviving_branch_id'),critique=d.get('feedback',{}).get('critique')))
            for bid,e in t['evaluations'].items():
                operations=[operation(v[bid]) for v in by_model.values() if bid in v]
                counts=Counter(operations)
                top=max(counts.values(),default=0)
                # Distinguish full 3/3 agreement from quorum and missing/discordant votes.
                agreement='3/3' if len(operations)==3 and top==3 else ('2+' if top>=2 else 'none')
                branches.append(dict(row,branch_id=bid,action=ACTION.get(e.get('decision'),'I'),
                    agreement=agreement,valid_votes=len(operations),
                    outcome=t['structural_action_outcomes'].get(bid),diagnostic=e.get('diagnostic')))
            for m1,m2 in itertools.combinations(MODELS,2):
                a,b=by_model.get(m1,{}),by_model.get(m2,{})
                for bid in a.keys() & b.keys():
                    pair_votes.append(dict(row,model1=m1,model2=m2,agree=operation(a[bid])==operation(b[bid])))
        runs.append(dict(identity,iterations=len(traces),converged=run['converged'],stop_reason=run['stop_reason'],
                         changed_iterations=changed_count,unchanged_iterations=unchanged_count,
                         exact_state_revisits=revisits,elapsed_seconds=run.get('elapsed_seconds')))
        snapshots.append(dict(name=name,config_id=config,run=run))

frames={name:pd.DataFrame(rows) for name,rows in dict(requests=requests,attempts=attempts,votes=votes,
    branches=branches,iterations=iterations,runs=runs,pair_votes=pair_votes).items()}
for name,df in frames.items():df.to_csv(OUT/(name+'.csv'),index=False)
req,att,vote,branch,it,run,pair=[frames[k] for k in ['requests','attempts','votes','branches','iterations','runs','pair_votes']]
assert len(run)==36 and len(req)==3*len(it)
assert set(vote.model)==set(MODELS)
assert branch.groupby(['dataset','config_id','iteration']).size().sum()==len(branch)
assert len(vote)==6372 and int(att.rejected.sum())==62
assert int((vote.action.eq('K') & vote.relationship.eq('NOT_EQUIVALENT')).sum())==157

plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10})
colors={'K':'#639d73','W':'#dcaa43','R':'#b04b4b','D':'#6484b4'}
fig,axes=plt.subplots(1,3,figsize=(10,3.6),sharey=True)
for ax,c in zip(axes,CONFIGS):
    bottom=np.zeros(3)
    for a in ['K','W','R','D']:
        values=[]
        for m in MODELS:
            v=vote[(vote.config_id==c)&(vote.model==m)]
            values.append(100*v.action.eq(a).mean())
        ax.bar(range(3),values,bottom=bottom,label=a,color=colors[a]);bottom+=values
    ax.set_xticks(range(3),list(MODELS.values()),rotation=30,ha='right')
    ax.set_title(LABELS[c]);ax.set_ylim(0,100)
axes[0].set_ylabel('Decisioni nelle risposte valide (%)')
handles,legend_labels=axes[-1].get_legend_handles_labels()
fig.legend(handles,legend_labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,1.02))
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'per_model_actions.pdf',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,3.4))
for c,color in zip(CONFIGS,['#2878a5','#dcaa43','#639d73']):
    rr=run[run.config_id==c]
    ax.step(range(26),[sum(not(r.converged and r.iterations<=t) for r in rr.itertuples()) for t in range(26)],where='post',label=LABELS[c],color=color)
ax.set(xlim=(0,25),ylim=(0,12.5),xlabel='Iterazioni completate',ylabel='Run senza arresto per conferma')
ax.set_yticks(range(0,13,2));ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(OUT/'confirmation_curves.pdf',bbox_inches='tight');plt.close(fig)

method=r'''\section{Protocollo operativo per similarit\`a e analisi del ciclo}
\label{sec:operational-experiment-protocol}
Questa sezione precisa come vengono prodotte e lette le osservazioni,
senza introdurre fattori aggiuntivi nel piano a cinque configurazioni.

\subsection{Dalle descrizioni alle etichette cosine}
Per ciascun dataset si conserva la baseline top-down e si raggruppano
gli LLG secondo il riferimento al padre. La ricostruzione riceve soltanto
le descrizioni dei figli non vuoti: non vede il padre, la ground truth
o il documento per riprodurli. Qwen restituisce una descrizione HLG$'$
in JSON; richieste e retry di ricostruzione non riutilizzano risposte
precedentemente scartate. I rami non ricostruibili sono registrati,
senza trasformare un dato mancante in similarit\`a zero.

MPNet codifica separatamente la descrizione ricostruita e quelle degli
HLG originali correnti, con vettori normalizzati. Il codice usa CPU,
cache delle rappresentazioni e nessun preprocessing lessicale aggiuntivo;
controlla testi vuoti, vettori non validi e superamento del limite di
token prima di ammettere il confronto. Per vettori unitari il prodotto
scalare coincide con la cosine:
\begin{equation}
u_i=\frac{e(HLG'_i)}{\|e(HLG'_i)\|},\quad
v_j=\frac{e(HLG_j)}{\|e(HLG_j)\|},\qquad s_{ij}=u_i^\top v_j.
\end{equation}
Ogni HLG$'_i$ viene confrontato con tutti gli HLG originali dello
stesso progetto: $j=i$ indica il proprio padre; $j\ne i$ un altro
padre. Non si usano confronti HLG$'$--HLG$'$ e non si incrociano dataset.
Con $\tau=0{,}68$, il confronto col padre produce
\texttt{EQUIVALENT} quando $s_{ii}\ge\tau$ e
\texttt{NOT\_EQUIVALENT} altrimenti. I confronti fuori diagonale
sopra soglia possono diventare candidati duplicati soltanto per HLG
dello stesso attore e con le disponibilit\`a richieste dal codice.
L'uguaglianza dell'attore usa il nome normalizzato per spazi e maiuscole;
non \`e una verifica semantica automatica dei ruoli.

Gli evaluator ricevono etichette fisse e coppie candidate testuali,
non i punteggi da riclassificare. La somiglianza numerica serve a
individuare confronti da verificare, mentre la documentazione determina
supporto e responsabilit\`a della funzionalit\`a. La soglia \`e comune
ai dataset e non viene adattata in base alla ground truth. Nello studio
esplorativo si ricalcolano i conteggi sopra soglia sulle stesse coppie
per diversi valori ipotizzati; non si eseguono altrettante pipeline
iterative. Il raggruppamento diretto di quel run, la sua copertura e
i valori osservati restano quelli descritti nella
Sezione~\ref{sec:cosine-tuning-all-datasets}.

\subsection{Esecuzione dei confronti di prompting e temperatura}
Ogni run riparte dalla baseline congelata, anzich\'e dall'output della
configurazione precedente. Il confronto P0--P1--P4 varia soltanto
il prefisso di scenari dimostrativi a temperatura 0,2; il confronto
T0--P4--T4 mantiene quattro scenari e varia la temperatura fra
0, 0,2 e 0,4. P4 \`e una sola condizione condivisa. Ricostruzione,
generazione e critic della riparazione mantengono le proprie impostazioni.
La verifica iniziale del backend controlla l'accettazione delle
temperature richieste; non dimostra identit\`a delle risposte a temperatura zero.
Una modifica di prompt o codice richiede una suite distinta e non viene
applicata retroattivamente ai dati storici. Le analisi del processo
di seguito riguardano P0, P1 e P4 della v111, tutte terminate sui dodici
dataset; i risultati parziali delle temperature non vi entrano.

\subsection{Unit\`a di conteggio per i singoli LLM e per il ciclo}
Si distinguono tre livelli: la risposta globale di un modello a una
iterazione; il tentativo tecnico usato per ottenerla; la decisione
di quel modello su un ramo all'interno della risposta accettata.
Un tentativo scartato non fornisce voti parzialmente utilizzabili.
La percentuale di azioni di un LLM ha come denominatore le sue decisioni
accettate, non il numero di richieste o di dataset. Le frequenze sono
quindi pesate dall'esposizione: run lunghe e gerarchie grandi contribuiscono
pi\`u delle run brevi. Non costituiscono accuratezza dei singoli modelli.

L'accordo confronta azione e, per una rimozione duplicata, superstite.
L'accordo a coppie usa solo rami per cui entrambi i modelli hanno una
risposta valida. Il quorum \`e misurato separatamente dall'applicazione
della proposta. Un voto di riscrittura non equivale a una riparazione
riuscita; \texttt{HLG\_LLG\_HANDOFF\_COMPLETED} registra il completamento
del passaggio della collezione, non il conteggio dei goal effettivamente
modificati. Pi\`u rami possono riferirsi alla medesima riparazione batch.

La convergenza \`e l'arresto per conferma di tutti i rami, mentre il
limite di 25 iterazioni \`e un diverso motivo di arresto. Per verificare
le modifiche si confrontano le collezioni HLG--LLG in ingresso e in
uscita della stessa iterazione. Le ripetizioni esatte di stato sono
rilevate mediante impronta del contenuto serializzato, preservando
l'ordine delle collezioni. Gli ID di ramo sono locali allo snapshot:
non vengono usati per sostenere che lo stesso goal oscilli dopo una
rigenerazione che potrebbe aver cambiato cardinalit\`a o ordine.
'''

results=r'''\section{Comportamento dei singoli evaluator e dinamica del bottom-up}
\label{sec:evaluator-dynamics-v111}
L'analisi riguarda le 36 run complete P0, P1 e P4 della suite v111,
alla temperatura comune 0,2 e soglia 0,68. Usa le tracce finali archiviate,
non nuove chiamate ai modelli. Le frequenze descrivono il comportamento
in questi input; non identificano un modello come corretto o superiore
in assenza di annotazioni indipendenti delle singole decisioni.
'''
results+=f'''Sono osservate {len(it)} iterazioni complete, {len(req)} risposte
globali di modello, {len(att)} tentativi tecnici e {len(branch)}
valutazioni aggregate di ramo. Le risposte valide sono {int(req.valid.sum())};
quelle definitivamente invalide sono {int((~req.valid).sum())}.
Un tentativo scartato e una risposta definitivamente invalida sono quindi
eventi differenti: una richiesta pu\\`o riuscire dopo un retry.
'''
results+=r'''\subsection{Distribuzione delle azioni per modello}
Nella Tabella~\ref{tab:model-action-dynamics}, K indica \texttt{KEEP},
W \texttt{REWRITE}, R \texttt{REMOVE} e D \texttt{DUPLICATE REMOVAL}.
Il totale N conta decisioni di ramo provenienti da risposte valide;
le colonne delle azioni sono conteggi, non percentuali di correttezza.
'''
rows=[]
for c in CONFIGS:
    for m in MODELS:
        v=vote[(vote.config_id==c)&(vote.model==m)]
        rows.append([LABELS[c],MODELS[m],len(v)]+[int(v.action.eq(a).sum()) for a in ['K','W','R','D']])
results+=table('Decisioni accettate per configurazione e singolo evaluator nella suite v111.',
 'tab:model-action-dynamics',['Config.','Modello','N','K','W','R','D'],rows,'llrrrrr')
results+=figure('per_model_actions','Distribuzioni percentuali delle azioni proposte dai modelli. W indica riscrittura e R rimozione. Le differenze non sono misure di accuratezza.','fig:model-action-dynamics')
results+=r'''Per interpretare i diversi totali occorre considerare anche le risposte
scartate e il numero di rami incontrati nelle run. Il confronto entro
configurazione espone i modelli agli stessi snapshot; fra configurazioni
le gerarchie possono divergere e generare un diverso carico di decisioni.
Una prevalenza di KEEP esprime una tendenza osservata a conservare i padri,
mentre una quota maggiore di REWRITE produce pi\`u proposte di intervento;
nessuna delle due frequenze dimostra, isolatamente, maggiore fedelt\`a documentale.
Nella condizione P4, Ministral propone 366 riscritture e 40 rimozioni
duplicate, contro 281 e 9 di Granite e 314 e 5 di Gemma. In P0 le
riscritture sono rispettivamente 222, 155 e 173. Ministral risulta quindi
pi\`u orientato all'intervento in questi due insiemi di snapshot, mentre
Granite propone pi\`u KEEP. In P1 la distribuzione \`e meno separata:
Gemma propone 222 riscritture, Ministral 218 e Granite 203.
Queste frequenze non vengono estese a input diversi dai casi osservati.
'''
rows=[]
for c in CONFIGS:
    for m in MODELS:
        q=req[(req.config_id==c)&(req.model==m)];a=att[(att.config_id==c)&(att.model==m)]
        rows.append([LABELS[c],MODELS[m],len(q),int(q.valid.sum()),len(a),int(a.rejected.sum())])
results+=table('Affidabilit\`a del formato globale e retry: richieste finali, risposte valide e tentativi scartati. Un tentativo scartato pu\`o essere seguito da una risposta valida.',
 'tab:model-retry-dynamics',['Config.','Modello','Richieste','Valide','Tentativi','Scartati'],rows,'llrrrr')
results+=r'''Gli errori registrati sono controlli tecnici o strutturali, non etichette
di scorrettezza semantica. In particolare, una risposta che non copre
esattamente gli ID richiesti viene scartata globalmente: gli altri modelli
non ereditano i suoi voti e non ne compensano i rami mancanti.
Tutti i 62 tentativi scartati di questo insieme riguardano Gemma:
7 in P0, 28 in P1 e 27 in P4. Ministral e Granite non registrano
tentativi scartati nelle run selezionate. Le sei risposte globali
definitivamente invalide non forniscono voti, mentre gli altri errori
sono recuperati entro il budget. Questo dato riguarda affidabilit\`a
del formato nel protocollo adottato, non qualit\`a documentale.

\paragraph{Risposta valida e rispetto delle istruzioni sono distinti.}
La regola KEEP soltanto con EQUIVALENT, riportata nel prompt corrente,
non \`e un vincolo esplicito di \path{_validate_decision}: il codice
controlla la relazione per REMOVE e la disponibilit\`a per REWRITE,
ma non respinge automaticamente KEEP con NOT\_EQUIVALENT.
La Tabella~\ref{tab:keep-relationship-audit} quantifica questi mantenimenti
nelle tracce v111. Si confronta una propriet\`a osservabile delle risposte
con il vincolo documentato; non si ricostruisce da un hash il testo
completo di un prompt storico non salvato. Le azioni rimangono quelle
effettivamente registrate, senza riclassificarle retroattivamente.
'''
rows=[]
for c in CONFIGS:
    cells=[]
    for m in MODELS:
        v=vote[(vote.config_id==c)&(vote.model==m)]
        cells.append(int((v.action.eq('K')&v.relationship.eq('NOT_EQUIVALENT')).sum()))
    rows.append([LABELS[c]]+cells)
results+=table('KEEP con relazione NOT\_EQUIVALENT nelle risposte accettate v111: audit dei dati, distinto dalla correttezza documentale.',
 'tab:keep-relationship-audit',['Config.','Ministral','Granite','Gemma'],rows)
results+=r'''Il conteggio mostra perch\'e la validit\`a del formato e la maggioranza
non vadano presentate come verifica completa delle condizioni del prompt.
Il codice corrente del validatore \`e riprodotto nel
Listing~\ref{lst:documentary-decision-validator}.
Anche gli altri criteri documentali, come la copertura funzionale del
superstite, richiedono il giudizio semantico e non sono dimostrati dal JSON.

\subsection{Accordo fra modelli e decisioni applicate}
'''
rows=[]
for c in CONFIGS:
    b=branch[branch.config_id==c]
    rows.append([LABELS[c],len(b),int(b.agreement.eq('3/3').sum()),int(b.agreement.eq('2+').sum()),
                 int(b.agreement.eq('none').sum()),int(b.action.eq('I').sum())])
results+=table('Accordo operativo per ramo: unanimit\`a dei tre modelli, quorum senza unanimit\`a, assenza di quorum ed esito finale inconcludente. Il superstite fa parte dell\'accordo.',
 'tab:quorum-dynamics',['Config.','Rami','3/3','Quorum 2','No quorum','Inconcl.'],rows)
results+=r'''L'assenza di quorum comprende sia il disaccordo tra voti validi sia
la mancanza di voti utilizzabili. Inoltre il quorum non esaurisce i
controlli: in P1 tre proposte di rimozione duplicata unanimi diventano
inconcludenti perch\'e il superstite non \`e confermato KEEP; lo stesso
accade in due casi P4 con maggioranza di due. Per questo il totale
degli esiti inconcludenti pu\`o superare quello dei soli rami senza quorum.
'''
rows=[]
for c in CONFIGS:
    for (m1,m2),p in pair[pair.config_id==c].groupby(['model1','model2']):
        rows.append([LABELS[c],MODELS[m1]+'--'+MODELS[m2],len(p),num(100*p.agree.mean())+r'\%'])
results+=table('Accordo a coppie sulle operazioni: denominatore limitato ai rami con entrambi i voti validi. Non viene calcolata una misura di correttezza.',
 'tab:pair-agreement-dynamics',['Config.','Coppia','Confronti','Accordo'],rows,'llrr')
outcome_labels={'KEEP_ORIGINAL_HIGH_LEVEL_GOAL':'KEEP', 'HLG_LLG_HANDOFF_COMPLETED':'Handoff batch',
 'REMOVED_WITH_CHILDREN':'Ramo rimosso','INCONCLUSIVE':'Inconcludente','REJECTED_STATE_PRESERVED':'Stato preservato'}
rows=[]
for c in CONFIGS:
    b=branch[branch.config_id==c]
    rows.append([LABELS[c]]+[int(b.outcome.eq(o).sum()) for o in outcome_labels])
results+=table('Esiti registrati dall\'orchestratore per ramo. Gli handoff possono riguardare la stessa generazione batch; non sono conteggi di chiamate o di goal riscritti.',
 'tab:applied-dynamics',['Config.','KEEP','Handoff','Rimossi','Inconcl.','Preservati'],rows).replace(
 r'\begin{table}[!htbp]',r'\begin{table}[H]')
results+=r'''L'accordo sull'azione non comporta accordo sul contenuto delle critiche.
Un esempio \`e Fitness Tracker, P4, iterazione 1, ramo 4. Tutti i modelli
propongono REWRITE, ma i feedback seguenti orientano diversamente il padre
\emph{Guide beginners through structured and progressive fitness plans}:
'''
case=documents['Fitness Tracker']['runs']['reference/repeat_001']['globalGoalCycle']['iterations'][0]
case_evidence=[]
for record in case['evaluator_responses']:
    d=next(d for d in record['decisions'] if d['branch_id']=='4')
    results+='\\paragraph{'+MODELS[record['model']]+'.}\n\\begin{quote}\\small\n'+esc(d.get('feedback',{}).get('critique',''))+'\n\\end{quote}\n'
    case_evidence.append(dict(dataset='Fitness Tracker',config_id='reference',iteration=1,model=record['model'],decision=d))
results+=r'''La maggioranza approva dunque l'intervento, senza risolvere automaticamente
la tensione fra restringere l'onboarding ai principianti ed estendere
adattabilit\`a e personalizzazione. Il generatore riceve le critiche concordi
sull'azione, anche quando la loro formulazione non \`e pienamente compatibile.
Questo caso spiega un limite concreto del quorum e del feedback concatenato;
non dimostra da solo la causa di tutte le variazioni di F1 del dataset.

\subsection{Convergenza, stabilit\`a dello stato e costo iterativo}
'''
rows=[]
for c in CONFIGS:
    rr=run[run.config_id==c];tt=it[it.config_id==c]
    rows.append([LABELS[c],f'{int(rr.converged.sum())}/12',int(rr.iterations.sum()),num(rr.iterations.mean(),2),
                 int(tt.changed.sum()),int((~tt.changed).sum()),int(tt.revisited_state.sum())])
results+=table('Arresto e modifiche nella suite v111. Le visite ripetute contano soltanto un ritorno esatto a uno stato gi\`a visto dopo una modifica; le iterazioni senza modifica sono separate.',
 'tab:state-dynamics',['Config.','Conv.','Iterazioni','Media','Modifica','Invariate','Ritorni'],rows)
rows=[]
for name in sorted(documents):
    cells=[]
    for c in CONFIGS:
        r=run[(run.dataset==name)&(run.config_id==c)].iloc[0]
        cells.append(str(r.iterations)+(r'$^{\dagger}$' if not r.converged else ''))
    rows.append([esc(SHORT.get(name,name))]+cells)
results+=table('Iterazioni per dataset e configurazione. Il simbolo $\dagger$ indica arresto al limite di 25 senza convergenza; gli altri valori indicano conferma di tutti i rami.',
 'tab:dataset-iteration-dynamics',['Dataset','P0','P1','P4'],rows)
results+=figure('confirmation_curves','Run non ancora arrestate per conferma dopo ciascuna iterazione. A 25 restano rappresentate le run terminate al limite: il loro completamento tecnico non equivale a convergenza.','fig:confirmation-dynamics')
results+=r'''Le iterazioni prive di cambiamenti non implicano necessariamente arresto:
possono rimanere riscritture che riproducono lo stesso testo, voti senza
quorum o aggiornamenti rifiutati. Viceversa, un handoff completato non
garantisce una modifica effettiva del contenuto. I ritorni esatti a stati
precedenti costituiscono un indicatore restrittivo di ciclicit\`a; una
serie di parafrasi differenti non viene classificata come lo stesso stato.
I soli conteggi aggregati delle azioni non dimostrano pertanto l'oscillazione
di un particolare goal.
In queste tracce non compaiono ritorni esatti a uno stato precedente
dopo una modifica. Si osservano per\`o 53 iterazioni senza variazioni
in P0, 38 in P1 e 26 in P4. La mancata convergenza non va quindi
descritta automaticamente come ciclo tra due gerarchie identiche:
pu\`o comprendere stasi o una successione di formulazioni differenti.
Nel riferimento P4 AUU e SIA 25--26 raggiungono il limite, mentre
Recycling converge solo alla ventiquattresima iterazione e Fitness
Tracker alla sedicesima.

La convergenza resta un criterio operativo e non una certificazione della
gerarchia. I risultati finali di F1 sono misurati anche per le run non
convergenti: eliminarle favorirebbe artificialmente i casi pi\`u semplici.
Le run lunghe producono molte osservazioni correlate dello stesso progetto,
che non diventano repliche indipendenti. Per questo le tabelle dei voti
usano conteggi di processo, mentre le metriche finali continuano ad
attribuire lo stesso peso ai dodici dataset.
'''

# Bind every reported local table to its machine-readable records.
method_sources=['src/bottom_up/goal_orchestrator.py','src/bottom_up/global_goal_evaluator.py',
                'src/bottom_up/goal_similarity.py','src/bottom_up/goal_reconstructor.py','src/llm_clients.py']
dump('evidence_manifest.json',dict(suite_id=SUITE.name,source_sha256=hashes,configurations=CONFIGS,
    method_source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in method_sources},
    runs=len(run),iterations=len(it),responses=len(req),attempts=len(att),branch_evaluations=len(branch),
    accepted_votes=len(vote),new_inference_calls=0,
    human_verification=dict(status='pending_author_review',verifier=None)))
dump('critique_case_evidence.json',case_evidence)
# Preserve final selected runs for reproducible audits even if archives are later edited.
snapshot_path=OUT/'selected_completed_runs.json'
dump('selected_completed_runs.json',snapshots)
(OUT/'experimental_process_protocol.tex').write_text(method,encoding='utf-8')
(OUT/'evaluator_dynamics_section.tex').write_text(results,encoding='utf-8')

def insert(text,key,anchor,body):
    start='% BEGIN '+key;end='% END '+key
    block=start+'\n'+body+'\n'+end+'\n\n'
    if start in text:
        a=text.index(start);b=text.index(end,a)+len(end)
        return text[:a]+block.rstrip()+text[b:]
    assert text.count(anchor)==1,anchor
    return text.replace(anchor,block+anchor)

main=ROOT/'main.tex';text=main.read_text(encoding='utf-8')
text=insert(text,'OPERATIONAL_EXPERIMENT_PROTOCOL',r'\section{Metriche e procedura di valutazione}',method)
text=insert(text,'EVALUATOR_DYNAMICS_V111',r'\section{Esecuzioni incluse e provenienza dei risultati}',results)
main.write_text(text,encoding='utf-8')
print(f'Added per-model and convergence analysis: {len(run)} completed runs, {len(it)} iterations, {len(vote)} accepted branch votes.')
print(run.groupby('config_id')[['iterations','converged','changed_iterations','unchanged_iterations','exact_state_revisits']].sum().to_string())
