"""Build a local, source-bound overview of completed experimental cohorts."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'output/evaluation/completed_experiments_20261026'
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {}


def read(path):
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def csv(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def esc(s):
    return ''.join({'&': r'\&', '_': r'\_', '%': r'\%', '#': r'\#', '$': r'\$'}.get(c, c) for c in str(s))


def num(x, signed=False):
    return format(float(x), '+.4f' if signed else '.4f').replace('.', ',')


def table(caption, label, heads, rows, columns=None):
    columns = columns or ('l' + 'r' * (len(heads) - 1))
    return '\n'.join([r'\begin{table}[htbp]', r'\centering\small',
        '\\caption{' + caption + '}\\label{' + label + '}',
        '\\begin{tabular}{' + columns + '}', r'\toprule',
        ' & '.join(heads) + r' \\', r'\midrule',
        *[' & '.join(map(str, row)) + r' \\' for row in rows],
        r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def figure(name, caption, label):
    return ('\\begin{figure}[htbp]\n\\centering\n'
            '\\includegraphics[width=\\textwidth]{output/evaluation/completed_experiments_20261026/'
            + name + '.pdf}\n\\caption{' + caption + '}\\label{' + label + '}\n\\end{figure}\n')


short = {'Assegno Unico Universale - SIA Project 24 25': 'AUU',
 'Ethical Purchasing Group - SIA Project 22 23': 'Ethical Purchasing',
 'Event Organization Portal - SIA Project 21 22': 'Event Organization',
 'La Reine Marlene - SIA Project 23 24': 'La Reine Marlene',
 'London Ambulance Service': 'London Ambulance', 'SIA Project 25 26': 'SIA 25--26'}
names = {'reference': 'P4', 'E1_0_examples': 'P0', 'E1_1_examples': 'P1',
 'E2_temperature_0.0': 'T0', 'E2_temperature_0.4': 'T4', 'E2_temperature_0.6': 'T6'}
suite_paths = sorted((ROOT / 'output/bottom_up_experiments').glob('*/manifest.json'))
inventory, cohorts, documents, manifests = [], [], {}, {}
for path in suite_paths:
    m = read(path)
    sid = path.parent.name
    manifests[sid] = m
    docs = [read(p) for p in sorted((path.parent / 'datasets').glob('*.json'))]
    documents[sid] = {d['name']: d for d in docs}
    counts = Counter(r.get('status') for d in docs for r in d.get('runs', {}).values())
    complete_configs = []
    for c in m['configurations']:
        runs = [d['runs'].get(f"{c['config_id']}/repeat_{i:03d}", {})
                for d in docs for i in range(1, m['repetitions'] + 1)]
        if len(docs) == 12 and all(r.get('status') == 'completed' for r in runs):
            complete_configs.append(c['config_id'])
            policies = {r['globalGoalCycle']['configuration']['pipeline_policy_version'] for r in runs}
            assert len(policies) == 1
            cohorts.append(dict(suite_id=sid, config_id=c['config_id'],
                examples=c['evaluator_few_shot_examples'], temperature=c['evaluator_temperature'],
                runs=len(runs), converged=sum(r.get('converged') is True for r in runs),
                iterations=np.mean([r['completed_iterations'] for r in runs]), policy=next(iter(policies))))
    inventory.append(dict(suite_id=sid, policy=m['fixed_runtime']['policy'], completed=counts['completed'],
                          other=sum(counts.values())-counts['completed'], complete_configs=complete_configs))

# Use each suite's latest exported final metrics, rather than mixing evaluation snapshots.
exports = sorted((ROOT / 'output/evaluation/configuration_comparison').glob('*/final_metrics.csv'))
exports += sorted((ROOT / 'output/evaluation/configuration_comparison').glob('*/metrics.csv'))
exports.sort(key=lambda p: p.parent.name)
evaluations = {}
for p in exports:
    d = pd.read_csv(p)
    for sid in d.suite_id.dropna().unique():
        evaluations[sid] = p

for c in cohorts:
    path = evaluations.get(c['suite_id'])
    if path is None:
        continue
    d = csv(path)
    final = d[(d.suite_id == c['suite_id']) & (d.config_id == c['config_id'])]
    for level in ['HLG', 'LLG']:
        a = final[final.level == level]
        if len(a) != c['runs'] or a.dataset.nunique() != 12:
            continue
        if not np.isfinite(a[['precision', 'recall', 'f1']].to_numpy()).all():
            continue
        base = d[(d.config_id == 'top_down') & (d.level == level)]
        paired = a.merge(base[['dataset', 'baseline_key', 'f1']], on=['dataset', 'baseline_key'],
                         suffixes=('', '_baseline'), validate='many_to_one')
        assert len(paired) == c['runs']
        c[f'f1_{level}'] = float(a.f1.mean())
        c[f'delta_{level}'] = float((paired.f1-paired.f1_baseline).mean())
    c['metric_source'] = str(path.relative_to(ROOT))

latest = 'ofat_fixed_thresholds_20261025_170859'
metrics_path = evaluations[latest]
metrics = csv(metrics_path)
metric_manifest = read(metrics_path.parent / 'evaluation_manifest.json')
selected = ['E1_0_examples', 'E1_1_examples', 'reference']
assert {c['config_id'] for c in cohorts if c['suite_id'] == latest} == set(selected)
current = metrics[metrics.config_id.isin(['top_down'] + selected)].copy()
assert len(current) == 96
notebook = read(ROOT / 'notebook/04_evaluation_all_configurations.ipynb')
evaluation_helpers = {'pd': pd}
exec(''.join(notebook['cells'][5]['source']), evaluation_helpers)
strict_comparison, strict_cohort, strict_completion = evaluation_helpers['completed_corpus_comparison_table'](
    metrics, csv(metrics_path.parent/'configurations.csv').to_dict('records'),
    csv(metrics_path.parent/'availability.csv').to_dict('records'), expected_datasets=12)
assert len(strict_cohort) == 12 and strict_comparison['Dataset valutati'].eq(12).all()
assert set(strict_comparison.Esperimento) == {'Top-down originale', *selected}
reported_cohorts = [c for c in cohorts if 'f1_HLG' in c and 'f1_LLG' in c]
assert all(c['runs'] == 12 for c in reported_cohorts)
for r in current.itertuples():
    doc = documents[latest][r.dataset]
    if r.config_id == 'top_down':
        goals = doc['baseline']['structuredHighLevelGoals' if r.level == 'HLG' else 'structuredLowLevelGoals']
    else:
        saved = doc['runs'][f'{r.config_id}/repeat_001']
        assert saved['status'] == 'completed'
        goals = saved['finalHighLevelGoals' if r.level == 'HLG' else 'finalLowLevelGoals']
    assert len(goals['goals' if r.level == 'HLG' else 'low_level_goals']) == r.generated_goal_count

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10})
macro = current.groupby(['config_id', 'level'])[['precision', 'recall', 'f1']].mean()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
labels = ['top_down'] + selected
for ax, level in zip(axes, ['HLG', 'LLG']):
    ax.bar(range(4), [macro.loc[(c, level), 'f1'] for c in labels], color=['#999999', '#2878a5', '#df9c34', '#55a176'])
    ax.set_xticks(range(4), ['Top-down', 'P0', 'P1', 'P4'])
    ax.set_title(level)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=.2)
axes[0].set_ylabel('F1 macro (12 dataset)')
fig.tight_layout()
fig.savefig(OUT / 'v111_macro_f1.pdf', bbox_inches='tight')
plt.close(fig)

base = current[current.config_id == 'top_down'].set_index(['dataset', 'level'])
dataset_names = sorted(documents[latest])
deltas = np.array([[current[(current.dataset == name) & (current.level == level) &
    (current.config_id == c)].iloc[0].f1 - base.loc[(name, level), 'f1']
    for c in selected for level in ['HLG', 'LLG']] for name in dataset_names])
fig, ax = plt.subplots(figsize=(10, 5))
limit = max(abs(deltas.min()), abs(deltas.max()))
im = ax.imshow(deltas, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
ax.set_yticks(range(12), [short.get(n, n).replace('--', '-') for n in dataset_names])
ax.set_xticks(range(6), [f'{names[c]} {l}' for c in selected for l in ['HLG', 'LLG']])
for i in range(12):
    for j in range(6):
        ax.text(j, i, f'{deltas[i,j]:+.3f}', ha='center', va='center', fontsize=8,
                color='white' if abs(deltas[i,j]) > limit*.6 else 'black')
fig.colorbar(im, ax=ax, label='Delta F1 rispetto alla baseline appaiata')
fig.tight_layout()
fig.savefig(OUT / 'v111_dataset_deltas.pdf', bbox_inches='tight')
plt.close(fig)

tuning_dir = ROOT / 'experiment_threhold_tuning/outputs/all_datasets_20261025_170409'
tuning = read(tuning_dir / 'cosine_summary.json')
pairs = csv(tuning_dir / 'cosine_pairs.csv')
parents = pairs[pairs.pair_type == 'prime_parent']
cross = pairs[(pairs.pair_type == 'prime_other_original') & pairs.same_actor]
assert len(parents) == 60 and len(cross) == 112
thresholds = [.50, .55, .60, .65, .68, .70, .75, .80, .85, .90]
grid = [(t, int((parents.cosine_similarity >= t).sum()), int((cross.cosine_similarity >= t).sum())) for t in thresholds]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].hist([parents.cosine_similarity, cross.cosine_similarity], bins=np.linspace(0, 1, 21),
             label=['HLG\N{PRIME}-padre (60)', 'Altri HLG, stesso attore (112)'], color=['#2878a5', '#df9c34'])
axes[0].axvline(.68, color='black', linestyle='--', label='Soglia 0,68')
axes[0].set_xlabel('Cosine similarity')
axes[0].set_ylabel('Numero di coppie')
axes[0].legend(fontsize=7)
axes[1].plot(thresholds, [a/60 for _, a, _ in grid], marker='o', label='Padri sopra soglia')
axes[1].plot(thresholds, [b/112 for _, _, b in grid], marker='s', label='Altri HLG sopra soglia')
axes[1].set_xlabel('Soglia ipotizzata')
axes[1].set_ylabel('Frazione delle rispettive coppie')
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / 'threshold_analysis.pdf', bbox_inches='tight')
plt.close(fig)

section = r'''\section{Quadro aggiornato degli esperimenti terminati}
\label{sec:completed-experiments-overview}
Questa sezione raccoglie i risultati finali disponibili negli archivi
esaminati il 26 ottobre 2026. Per i confronti sul corpus si includono
soltanto configurazioni terminate su tutti e dodici i dataset e su tutte
le ripetizioni previste. Una run con stato \texttt{completed} resta
inclusa anche se termina al limite iterativo senza convergere.
Gli stati \texttt{running}, \texttt{interrupted} e \texttt{not\_started}
non sono risultati finali e non sono sostituiti da zeri.

\subsection{Inventario delle esecuzioni e confronti ammessi}
Le versioni del prompt sono mantenute separate: cambiamenti di policy,
input o temperatura non consentono di interpretare le differenze fra
suite come effetto causale di un singolo fattore. Ogni voce dell'inventario
conta una run salvata, non una replica necessariamente indipendente;
eventuali riusi non costituiscono nuova evidenza sperimentale.
Nelle tabelle seguenti compaiono soltanto configurazioni concluse sui
dodici dataset e con metriche finali disponibili per entrambi i livelli.
Le configurazioni parziali non compaiono neppure nei conteggi riportati
in questo capitolo e non restringono la coorte delle medie.
'''
rows=[]
for item in inventory:
    sid=item['suite_id']; stamp=sid.removeprefix('ofat_fixed_thresholds_')
    included = [c for c in reported_cohorts if c['suite_id'] == sid]
    if not included:
        continue
    version=item['policy'].split('_v')[1].split('_')[0]
    rows.append([stamp.replace('_', r'\_'), 'v'+version, sum(c['runs'] for c in included),
                 ', '.join(names[c['config_id']] for c in included)])
section += table('Configurazioni incluse nei risultati: ogni configurazione comprende dodici run finali sui dodici dataset. Il numero di run include esclusivamente le configurazioni riportate, senza esecuzioni parziali.',
 'tab:completed-suite-inventory', ['Suite (data e ora)', 'Policy', 'Run incluse', 'Configurazioni complete'], rows, 'llrl')
section += r'''P0 e P1 indicano zero-shot e one-shot; P4 indica il riferimento
few-shot configurato nella relativa suite. T0, T4 e T6 identificano le
temperature 0,0, 0,4 e 0,6 del piano storico. Le impostazioni effettive
del riferimento sono riportate nella tabella successiva: non si assume
che siano identiche al piano corrente a temperatura 0,2.
Le suite senza metriche finali complete e verificate non vengono
riportate come risultati quantitativi.
'''
rows=[]
for c in reported_cohorts:
    version='v'+c['policy'].split('_v')[1].split('_')[0]
    stamp=c['suite_id'].removeprefix('ofat_fixed_thresholds_')
    rows.append([version, names[c['config_id']], str(c['examples']), str(c['temperature']).replace('.',','),
                 num(c['f1_HLG']) if 'f1_HLG'in c else '--', num(c['f1_LLG']) if 'f1_LLG'in c else '--',
                 f"{c['converged']}/{c['runs']}"])
section += table('Risultati definitivi delle configurazioni complete: esempi, temperatura, F1 macro e convergenza. Ogni riga comprende dodici dataset con output finali valutati; le suite sono confrontate separatamente.',
 'tab:completed-all-cohorts', ['Policy', 'Config.', 'Esempi', 'T', 'F1 HLG', 'F1 LLG', 'Convergenti'],rows)
section += r'''Le metriche provengono dagli export finali del notebook 04,
associati alla rispettiva suite; gli hash e i percorsi sono registrati
negli artefatti di questa analisi. Non si aggregano esportazioni ripetute
delle stesse run. I risultati storici v103 gi\`a discussi nelle sezioni
successive restano riferiti al loro snapshot originale; la tabella
qui sopra include anche T0, completata successivamente a quello snapshot.

\subsection{Risultati della suite v111 a temperatura 0,2}
La suite \path{ofat_fixed_thresholds_20261025_170859} usa gli input
schematici v111 e la soglia 0,68. P0, P1 e P4 hanno ciascuna dodici
run finali e condividono le stesse baseline congelate. Queste sono le
sole configurazioni della suite incluse nei risultati definitivi.
Le configurazioni di temperatura non ancora complete sono escluse
interamente, senza riportarne medie o conteggi parziali.
Le metriche vengono ricalcolate come medie aritmetiche delle righe
finali esportate, selezionando esplicitamente i dodici dataset comuni
a P0, P1 e P4. Si applica lo stesso filtro della sezione del notebook 04
dedicata alle configurazioni concluse su tutti i dodici dataset:
le configurazioni parziali non determinano l'intersezione del corpus.
Il matching usa
\texttt{bert-base-uncased}, le descrizioni dei goal e il preprocessing
NLTK; non applica la soglia MPNet 0,68 alla valutazione esterna.
La baseline \`e appaiata tramite dataset e identificativo della baseline.
'''
rows=[]
for c in ['top_down']+selected:
    for level in ['HLG','LLG']:
        r=macro.loc[(c,level)]
        delta=r.f1-macro.loc[('top_down',level),'f1']
        method = 'Top-down originale' if c == 'top_down' else c
        checked = strict_comparison.set_index('Esperimento').loc[method]
        assert abs(checked[f'{level} F1'] - r.f1) < 1e-12
        rows.append(['Top-down' if c=='top_down' else names[c],level,12,num(r.precision),num(r.recall),num(r.f1),num(delta,True)])
section += table('Suite v111: Precision, Recall e F1 macro sui dodici dataset, con differenza assoluta di F1 dalla baseline appaiata. Una ripetizione per configurazione.',
 'tab:v111-final-macro', ['Metodo','Livello','Dataset','Precision','Recall','F1',r'$\Delta$F1'],rows)
section += figure('v111_macro_f1','F1 macro della suite v111 e delle baseline. La scala comune 0--1 evita di amplificare visivamente differenze piccole.','fig:v111-macro')
section += f'''P0 ottiene F1 HLG {num(macro.loc[('E1_0_examples','HLG'),'f1'])} e
F1 LLG {num(macro.loc[('E1_0_examples','LLG'),'f1'])}, i valori maggiori
fra le tre configurazioni di questa suite. P4 ottiene invece F1 HLG
{num(macro.loc[('reference','HLG'),'f1'])}, sotto la baseline
{num(macro.loc[('top_down','HLG'),'f1'])}, e F1 LLG
{num(macro.loc[('reference','LLG'),'f1'])}, poco sopra la baseline
{num(macro.loc[('top_down','LLG'),'f1'])}. L'aggiunta di esempi non produce
quindi un miglioramento monotono. Le differenze sono descrittive:
una sola generazione per coppia dataset--configurazione non permette
di stimare stabilit\\`a o significativit\\`a dell'ordinamento.
'''
rows=[]
for i,name in enumerate(dataset_names):
    rows.append([esc(short.get(name,name))]+[num(x,True) for x in deltas[i]])
section += table('Suite v111: differenze di F1 per dataset rispetto al proprio top-down. H e L indicano HLG e LLG.',
 'tab:v111-all-dataset-deltas', ['Dataset','P0 H','P0 L','P1 H','P1 L','P4 H','P4 L'],rows)
section += figure('v111_dataset_deltas','Variazioni di F1 per dataset nella suite v111. Blu indica aumento e rosso diminuzione rispetto alla stessa baseline; ogni cella conserva il valore numerico.','fig:v111-deltas')
section += r'''I risultati non si muovono sempre insieme ai due livelli.
Nel riferimento P4, SIA 25--26 migliora soprattutto negli LLG
($\Delta$F1 = +0,1624), mentre London Ambulance peggiora negli LLG
($\Delta$F1 = -0,1699) pur migliorando leggermente negli HLG.
Calculator, Ethical Purchasing, Genome Nexus e Gestao Hospital
mantengono in P4 gli stessi punteggi della baseline. Una media vicina
alla baseline pu\`o quindi nascondere miglioramenti e peggioramenti
consistenti su singoli progetti. Una maggiore specificit\`a del testo
non coincide automaticamente con maggiore copertura o correttezza.

\subsection{Esempi di goal finali e interpretazione delle trasformazioni}
Gli estratti seguenti provengono dagli output finali \texttt{reference}
della suite v111, non dalle dimostrazioni fittizie del prompt.
La selezione comprende un miglioramento marcato, un peggioramento,
un effetto diverso tra livelli e un caso invariato; non rappresenta
un campionamento casuale n\'e una verifica manuale esaustiva dei requisiti.
'''
case_records=[]
for dataset,needle,explanation in [
 ('SIA Project 25 26','Preliminary',r'''La formulazione finale restringe la responsabilit\`a dell'ufficio alla verifica preliminare e rende esplicita la motivazione del rifiuto, invece di attribuirgli l'intero processo fino alla risoluzione. Questa trasformazione illustra una correzione dei confini di responsabilit\`a; l'aumento di F1 degli LLG riguarda l'intero dataset e non pu\`o essere attribuito causalmente a questo solo padre.'''),
 ('London Ambulance Service','',r'''Il dataset presenta un piccolo aumento di F1 HLG ma una diminuzione marcata di F1 LLG. L'esempio mostra il contenuto di un padre finale; il peggioramento quantitativo della decomposizione non dimostra, da solo, che ogni sua frase sia errata. Cambiamenti di granularit\`a, cardinalit\`a e corrispondenza con il riferimento possono contribuire al punteggio: stabilire il peso di ciascun fattore richiede un'analisi delle singole corrispondenze.'''),
 ('Event Organization Portal - SIA Project 21 22','Offer',r'''Il padre finale esplicita gestione delle richieste di appuntamento, preventivi e pagamenti associati agli appuntamenti confermati. Il risultato complessivo aumenta F1 LLG (+0,0723), ma riduce F1 HLG (-0,0263): una descrizione pi\`u dettagliata e una decomposizione pi\`u vicina al riferimento possono coesistere con una minore corrispondenza al livello superiore.'''),
 ('Calculator','',r'''Il padre rimane identico alla baseline e i punteggi HLG e LLG non cambiano. La revisione pu\`o quindi confermare un risultato gi\`a presente; assenza di modifica e assenza di miglioramento non equivalgono a un fallimento tecnico.'''),
]:
    doc=documents[latest][dataset];run=doc['runs']['reference/repeat_001']
    goals=run['finalHighLevelGoals']['goals']
    goal=next((g for g in goals if needle and needle in g['name']),goals[0])
    original=next((g for g in doc['baseline']['structuredHighLevelGoals']['goals'] if g['actor']['name']==goal['actor']['name']),doc['baseline']['structuredHighLevelGoals']['goals'][0])
    section+='\\paragraph{'+esc(short.get(dataset,dataset))+'.}\n'
    section+='Attore: '+esc(goal['actor']['name'])+'. Padre iniziale: \\emph{'+esc(original['name'])+'}.\n'
    section+='\\begin{quote}\\small\n\\textbf{'+esc(goal['name'])+'}\\\\\n'+esc(goal['description'])+'\n\\end{quote}\n'+explanation+'\n'
    section+='La descrizione iniziale era: \\emph{'+esc(original['description'])+'}.\n'
    case_records.append(dict(dataset=dataset,config_id='reference',repetition=1,original=original,final=goal,explanation=explanation))

section += r'''\subsection{Esperimento esplorativo sulla soglia cosine}
\label{sec:completed-threshold-results}
L'esperimento di soglia \`e distinto dalle run iterative precedenti:
ricostruisce HLG$'$ dalle baseline dei dodici dataset e calcola i
punteggi delle coppie, senza rieseguire l'intera pipeline per ogni
valore ipotizzato. Sono disponibili 60 ricostruzioni rispetto a 61
padri iniziali: un padre privo di figli non viene ricostruito.
Le 394 coppie comprendono 60 confronti col padre e 334 con altri
HLG del progetto; fra questi ultimi, 112 riguardano lo stesso attore.
La media cosine HLG$'$--padre \`e 0,692983 e la media delle medie
per dataset \`e 0,697144. Sono statistiche di similarit\`a testuale,
non misure di accuratezza delle decisioni degli evaluator.
Le motivazioni, la copertura per dataset e tutti i punteggi sono
documentati nella Sezione~\ref{sec:cosine-tuning-all-datasets} e
nell'appendice dei valori completi.
'''
section += table('Analisi di sensibilit\`a sulle stesse coppie: numero di padri sopra soglia e di altri HLG dello stesso attore sopra soglia. Non sono dieci run iterative n\'e conteggi di veri duplicati.',
 'tab:completed-threshold-grid', ['Soglia','Padri sopra soglia / 60','Altri HLG sopra soglia / 112'],
 [[str(t).replace('.',','),str(a),str(b)] for t,a,b in grid])
section += figure('threshold_analysis','Distribuzione dei punteggi e sensibilit\`a dello screening alla soglia. Le frazioni a destra hanno denominatori diversi (60 e 112); non rappresentano Recall o tasso di falsi positivi.','fig:completed-threshold-analysis')
section += r'''Una soglia pi\`u bassa aumenta sia le relazioni considerate equivalenti
sia i candidati duplicati da verificare; una soglia pi\`u alta restringe
entrambi gli insiemi. Il valore operativo 0,68 viene quindi descritto
come scelta esplorativa sulla distribuzione osservata. Senza annotazioni
indipendenti delle coppie non \`e possibile affermare che massimizzi
Precision, Recall o qualit\`a della gerarchia finale.
La tabella rende osservabile la sensibilit\`a, ma non dimostra
che uno dei valori sia ottimale.
'''
# Examples nearest the operative cutoff, with real source texts and scores.
below=parents[parents.cosine_similarity < .68].nlargest(1, 'cosine_similarity')
above=parents[parents.cosine_similarity >= .68].nsmallest(1, 'cosine_similarity')
cross_near=cross.iloc[(cross.cosine_similarity-.68).abs().argsort()].head(1)
near=pd.concat([below, above, cross_near])
for row in near.itertuples():
    section+='\\paragraph{Coppia reale: '+esc(short.get(row.dataset,row.dataset))+'.}\n'
    section+='HLG$\'$: \\emph{'+esc(row.hlg_prime)+'}. HLG confrontato: \\emph{'+esc(row.original_hlg_name)+'}; descrizione: \\emph{'+esc(row.original_hlg_description)+'}. '
    section+='Cosine = '+num(row.cosine_similarity)+'; confronto '+('col padre' if row.pair_type=='prime_parent' else 'con un altro HLG')+'. '
    section+=('Il punteggio supera' if row.cosine_similarity>=.68 else 'Il punteggio non raggiunge')+' la soglia 0,68; tale esito numerico non sostituisce il controllo documentale.\n'
section += r'''\subsection{Sintesi dei risultati ottenuti}
Le esecuzioni concluse mostrano che la revisione pu\`o modificare
copertura, responsabilit\`a e granularit\`a dei goal, oppure lasciare
intatta la baseline. Nella suite v111, P0 ottiene la media maggiore
fra le configurazioni complete, mentre P4 produce guadagni e perdite
molto diversi tra dataset. Gli esempi nel prompt sono quindi una guida
decisionale da valutare, non una garanzia di miglioramento.
L'analisi cosine descrive quali coppie vengono sottoposte a tale giudizio;
non valida autonomamente la correttezza delle azioni. Le conclusioni
restano circoscritte alle run terminate, a una ripetizione e al matching
testuale adottato. Le analisi storiche che seguono conservano il loro
perimetro e non vengono reinterpretate come risultati della v111.
'''

(OUT/'completed_experiments_section.tex').write_text(section,encoding='utf-8')
pd.DataFrame(reported_cohorts).to_csv(OUT/'complete_cohorts.csv',index=False)
strict_comparison.to_csv(OUT/'v111_complete_comparison_12.csv',index=False)
strict_completion.to_csv(OUT/'v111_configuration_completion_12.csv',index=False)
current.to_csv(OUT/'v111_final_metrics.csv',index=False)
pd.DataFrame(grid,columns=['threshold','parent_above','same_actor_other_above']).to_csv(OUT/'threshold_grid.csv',index=False)
(OUT/'case_evidence.json').write_text(json.dumps(case_records,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'evidence_manifest.json').write_text(json.dumps(dict(
    captured_on='2026-10-26',sources=SOURCES,inventory=inventory,
    cohort_policy='All 12 datasets and all planned repetitions completed; both final metric levels available and finite; no partial cohort rows reported',
    reported_complete_cohorts=len(reported_cohorts), reported_runs=sum(c['runs'] for c in reported_cohorts),
    current_metric_source=str(metrics_path.relative_to(ROOT)),
    current_cohort_datasets=dataset_names,metric_definition=metric_manifest['metric'],
    verified_generated_goal_counts=True,new_inference_calls=0,
    human_verification={'status':'pending_author_review','verifier':None}),ensure_ascii=False,indent=2),encoding='utf-8')

main_path=ROOT/'main.tex';text=main_path.read_text(encoding='utf-8')
start='% BEGIN COMPLETED_EXPERIMENTS_OVERVIEW';end='% END COMPLETED_EXPERIMENTS_OVERVIEW'
block=start+'\n'+section+'\n'+end+'\n\n'
if start in text:
    left=text.index(start);right=text.index(end,left)+len(end)
    text=text[:left]+block.rstrip()+text[right:]
else:
    anchor=r'\section{Esecuzioni incluse e provenienza dei risultati}'
    assert text.count(anchor)==1
    text=text.replace(anchor,block+anchor)
text=text.replace('Questo capitolo riporta i risultati della suite',
                  'Le sezioni storiche seguenti riportano i risultati della suite')
text=text.replace('I risultati congelati in questo capitolo riguardano',
                  'I risultati dello snapshot storico riguardano')
conclusion_start=text.index("L'implementazione conserva la baseline", text.index(r'\chapter{Conclusion and Future Work}'))
conclusion_end=text.index('Gli sviluppi successivi comprendono',conclusion_start)
text=text[:conclusion_start]+r'''L'implementazione conserva la baseline, registra gli snapshot e separa
i segnali usati durante la revisione dalle metriche esterne contro la
ground truth. L'inventario aggiornato nella
Sezione~\ref{sec:completed-experiments-overview} distingue le versioni
storiche dai confronti terminati della suite v111. In quest'ultima,
zero-shot ottiene le maggiori medie F1 sia HLG sia LLG tra le tre
configurazioni complete; one-shot e few-shot non migliorano F1 HLG
rispetto al top-down, pur aumentando F1 LLG in media. Le differenze
fra dataset restano ampie e una sola ripetizione non dimostra un
vantaggio generale del prompting senza esempi. L'analisi storica v103
evidenzia a sua volta comportamenti diversi, che non vengono aggregati
con quelli della v111. Le temperature della suite pi\`u recente non
sono ancora complete sul corpus; i risultati delle configurazioni
storiche terminate sono riportati con il proprio contesto.

L'esperimento sulla soglia rende espliciti i punteggi osservati e la
sensibilit\`a dello screening ai valori considerati, ma non dimostra
l'ottimalit\`a di 0,68. Nel complesso, i casi osservati mostrano che
accordo degli evaluator, convergenza e maggiore dettaglio dei testi
non garantiscono la correttezza della gerarchia. Metriche, goal finali
e tracce decisionali devono quindi essere interpretati insieme.

'''+text[conclusion_end:]
main_path.write_text(text,encoding='utf-8')
print(f'Included {len(reported_cohorts)} fully evaluated complete cohorts; {len(current)} current final metric rows; 3 figures; 4 real goal examples.')
