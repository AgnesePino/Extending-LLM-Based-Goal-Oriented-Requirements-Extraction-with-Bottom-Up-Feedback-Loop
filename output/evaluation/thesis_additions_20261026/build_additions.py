"""Generate thesis additions exclusively from frozen local numerical evidence."""
import hashlib
import inspect
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TUNING = ROOT / 'experiment_threhold_tuning/outputs/all_datasets_20261025_170409'
RESULTS = ROOT / 'output/evaluation/thesis_completed_v103_20261025'


def load(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def escape(value):
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
               '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
               '^': r'\textasciicircum{}'}
    return ''.join(mapping.get(c, c) for c in str(value))


def number(value, digits=4, signed=False):
    return format(float(value), f'{"+" if signed else ""}.{digits}f').replace('.', ',')


def table(caption, label, columns, headings, rows):
    return '\n'.join([
        r'\begin{table}[htbp]', r'\centering', r'\small',
        '\\caption{' + caption + '}', '\\label{' + label + '}',
        '\\begin{tabular}{' + columns + '}', r'\toprule',
        ' & '.join(headings) + r' \\', r'\midrule',
        *[' & '.join(row) + r' \\' for row in rows],
        r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def long_table(caption, label, columns, headings, rows):
    header = ' & '.join(headings) + r' \\'
    return '\n'.join([
        r'\begingroup', r'\small', r'\setlength{\tabcolsep}{3pt}',
        '\\begin{longtable}{' + columns + '}',
        '\\caption{' + caption + '}\\label{' + label + r'}\\',
        r'\toprule', header, r'\midrule', r'\endfirsthead',
        r'\toprule', header, r'\midrule', r'\endhead',
        r'\bottomrule', r'\endfoot',
        *[' & '.join(row) + r' \\' for row in rows],
        r'\end{longtable}', r'\endgroup', ''])


snapshot = load(RESULTS / 'source_snapshot.json')
metrics = pd.read_csv(RESULTS / 'final_metrics.csv')
comparison = pd.read_csv(RESULTS / 'dataset_comparison.csv')
tuning = load(TUNING / 'cosine_summary.json')
review = load(TUNING / 'parent_review.json')
pairs = pd.read_csv(TUNING / 'cosine_pairs.csv')
by_dataset = pd.read_csv(TUNING / 'cosine_by_dataset.csv')
assert digest(TUNING / 'parent_review.json') == tuning['source_sha256']
assert digest(RESULTS / 'source_snapshot.json') == load(RESULTS / 'evidence_manifest.json')['source_snapshot_sha256']
assert len(pairs) == 394 and len(snapshot['documents']) == 12
parents = pairs[pairs.pair_type == 'prime_parent'].copy()
cross = pairs[pairs.pair_type == 'prime_other_original'].copy()
assert len(parents) == 60 and len(cross) == 334
assert abs(parents.cosine_similarity.mean() - tuning['overall']['prime_parent']['mean']) < 1e-12
assert abs(parents.groupby('dataset').cosine_similarity.mean().mean() - tuning['overall']['prime_parent']['macro_dataset_mean']) < 1e-12

short = {
    'Assegno Unico Universale - SIA Project 24 25': 'AUU',
    'Calculator': 'Calculator',
    'Ethical Purchasing Group - SIA Project 22 23': 'Ethical Purchasing',
    'Event Organization Portal - SIA Project 21 22': 'Event Organization',
    'Fitness Tracker': 'Fitness Tracker', 'Genome Nexus': 'Genome Nexus',
    'Gestao Hospital': 'Gestao Hospital',
    'La Reine Marlene - SIA Project 23 24': 'La Reine Marlene',
    'London Ambulance Service': 'London Ambulance', 'Planning Poker': 'Planning Poker',
    'Recycling': 'Recycling', 'SIA Project 25 26': 'SIA 25--26',
}

calibration = r'''\subsection{Analisi esplorativa della soglia su tutti i dataset}
\label{sec:cosine-tuning-all-datasets}
% Evidence C-T01/C-T02: cosine_summary.json, cosine_pairs.csv, parent_review.json.
Per esaminare il valore operativo della soglia è stata eseguita un'analisi
descrittiva sui dodici dataset, distinta dagli esperimenti che variano esempi
e temperatura. A partire dalle baseline top-down sono state generate nuove
ricostruzioni con Qwen3-VL-8B e lo stesso prompt, usando soltanto le descrizioni
degli LLG associati. MPNet ha confrontato ogni HLG$'$ con la descrizione del
proprio padre e con quelle degli altri HLG dello stesso progetto. I nomi dei
goal e le ground truth non entrano in questo calcolo.

Il corpus comprende 61 HLG, dei quali 60 hanno figli associabili e producono
una ricostruzione. Un ramo di SIA 25--26 non è ricostruibile; quattro LLG,
due in Recycling e due in SIA 25--26, non sono associati dal raggruppamento
usato in questa analisi. Questi casi sono registrati, senza assegnare loro
una cosine pari a zero. Sono disponibili 60 confronti con il padre e 334
confronti direzionali con altri HLG; 112 di questi ultimi riguardano lo stesso
attore. Non vengono confrontate due ricostruzioni.
Questo run usa il raggruppamento diretto dei riferimenti ai padri, senza il
recupero dei riferimenti non corrispondenti usato in alcune versioni del ciclo
completo; la copertura dichiarata si riferisce a tale procedura.

\paragraph{Distribuzione dei valori osservati.}
La media dei 60 confronti HLG$'$--padre è 0,69298, la mediana è 0,69766
e la deviazione standard della popolazione osservata è 0,11064; minimo
e massimo sono rispettivamente 0,43849 e 0,88138. La media delle dodici
medie per dataset è 0,69714. La prima media assegna lo stesso peso a ogni
ramo, mentre la seconda assegna lo stesso peso a ogni progetto.
I confronti con altri HLG hanno media 0,43388; limitandoli allo stesso attore
la media è 0,42825. Queste popolazioni hanno scopi differenti e non vengono
mescolate per ricavare un unico valore medio.

'''
rows = []
for name in short:
    a = parents[parents.dataset == name].cosine_similarity
    rows.append([short[name], str(len(a)), number(a.mean(), 5), number(a.min(), 5), number(a.max(), 5)])
calibration += table('Cosine HLG$\prime$--padre nei dodici dataset: tutte le ricostruzioni disponibili.',
                     'tab:cosine-tuning-datasets', 'lrrrr',
                     ['Dataset', '$n$', 'Media', 'Minimo', 'Massimo'], rows)

# This is a descriptive sweep: no equivalence truth labels, accuracy or optimality.
grid_rows = []
eligibility = {(d['dataset'], str(p['branch_id'])): p['status'] == 'completed'
               for d in review['datasets'] for p in d['pairs']}
eligible_cross = cross[cross.apply(lambda r: bool(r.same_actor) and
                                  eligibility[(r.dataset, str(r.other_branch_id))], axis=1)]
for threshold in [.50, .55, .60, .65, .68, .70, .75, .80, .85, .90]:
    grid_rows.append([number(threshold, 2), str(int((parents.cosine_similarity >= threshold).sum())),
                      str(int((parents.cosine_similarity < threshold).sum())),
                      str(int((eligible_cross.cosine_similarity >= threshold).sum()))])
calibration += r'''\paragraph{Confronto descrittivo tra soglie.}
La Tabella~\ref{tab:cosine-threshold-sensitivity} mostra come cambiano le
etichette e il numero di coppie candidate applicando diversi valori agli
stessi punteggi salvati. Non sono state rieseguite le pipeline per ogni soglia.
I conteggi non sono veri o falsi positivi: mancano annotazioni indipendenti
di equivalenza funzionale e quindi non misurano accuratezza o F1 del filtro.
'''
calibration += table('Sensibilità descrittiva delle classificazioni alla soglia. Le coppie candidate sono direzionali e dello stesso attore.',
                     'tab:cosine-threshold-sensitivity', 'rrrr',
                     [r'$\tau$', 'Padri sopra soglia', 'Padri sotto soglia', 'Coppie candidate'], grid_rows)
calibration += r'''\paragraph{Interpretazione della scelta operativa.}
La soglia 0,68 è una scelta operativa basata su una lettura d'insieme dei
confronti, non sulla regola ``soglia uguale alla media''. Con questo valore,
34 confronti con il padre risultano sopra soglia e 26 sotto; cinque coppie
direzionali dello stesso attore diventano candidate alla verifica dei duplicati.
La vicinanza alla media non dimostra che 0,68 sia ottimale: funzionalità
complementari possono avere cosine elevate e una formulazione valida può
avere una ricostruzione più astratta con cosine inferiore.

L'analisi qui riportata è successiva alle esecuzioni storiche del capitolo
dei risultati, nelle quali 0,68 era già stato adottato. Offre quindi un
riscontro descrittivo alla scelta, non una validazione retroattiva o una
calibrazione supervisionata. Non risultano annotazioni manuali di equivalenza
complete; una selezione statistica della soglia richiederebbe tali annotazioni
e una verifica su progetti separati.

\paragraph{Esempi reali.}
La Tabella~\ref{tab:cosine-real-examples} riporta testi effettivamente
confrontati e i punteggi ottenuti. AUU presenta sia il valore minimo sia un
confronto appena sotto 0,68; SIA 25--26 e Planning Poker mostrano casi appena
sopra soglia. Il confronto incrociato di Recycling illustra perché una
cosine elevata non basta a giustificare una rimozione: ricerca dei centri,
mappe e notifiche possono condividere termini ma includere risultati distinti.
'''
selections = [
    parents.nsmallest(1, 'cosine_similarity').iloc[0],
    parents.nlargest(1, 'cosine_similarity').iloc[0],
    parents[(parents.dataset.str.startswith('Assegno')) & (parents.branch_id == 3)].iloc[0],
    parents[(parents.dataset == 'SIA Project 25 26') & (parents.branch_id == 6)].iloc[0],
    parents[(parents.dataset == 'Planning Poker') & (parents.branch_id == 3)].iloc[0],
    cross[(cross.dataset == 'Recycling') & (cross.branch_id == 1) & (cross.other_branch_id == 3)].iloc[0],
]
example_rows = [[short[r.dataset] + ' ' + str(r.branch_id) + r'$\rightarrow$' + str(r.other_branch_id),
                 escape(r.hlg_prime), escape(r.original_hlg_description), number(r.cosine_similarity, 5)]
                for r in selections]
calibration += long_table('Esempi reali di confronti cosine. I testi sono riportati dai file del tuning; l’ultima riga è un confronto con un altro padre.',
                          'tab:cosine-real-examples',
                          r'@{}p{0.17\textwidth}p{0.33\textwidth}p{0.33\textwidth}r@{}',
                          ['Dataset e coppia', 'HLG$\prime$', 'Descrizione HLG', 'Cosine'], example_rows)
calibration += r'''L'Appendice~\ref{app:all-cosine-values} elenca tutti i 394 punteggi,
senza selezionare soltanto quelli vicini alla soglia. I CSV ed Excel del run
\path{all_datasets_20261025_170409} conservano anche i testi completi,
gli identificativi e la copertura dei dataset.
'''
(OUT / 'cosine_tuning_section.tex').write_text(calibration, encoding='utf-8')

appendix = r'''\chapter{Valori completi dell'analisi cosine}
\label{app:all-cosine-values}
% Evidence C-T02: cosine_pairs.csv, all 394 rows, no manual filtering.
I valori sono quelli del run \path{all_datasets_20261025_170409}.
Ogni riga identifica il dataset, il ramo di HLG$'$ ($i$), il padre confrontato
($j$), il tipo di confronto e la cosine. P indica il proprio padre; A un altro
HLG. Gli identificativi sono locali a ciascun progetto e coincidono con quelli
di \path{parent_review.json}. I nomi abbreviati identificano i dodici progetti
elencati nel corpus. Il confronto è direzionale: le righe $i\rightarrow j$ e
$j\rightarrow i$, quando presenti, non sono duplicati dello stesso calcolo.

'''
all_rows = [[short[r.dataset], str(r.branch_id), str(r.other_branch_id),
             'P' if r.pair_type == 'prime_parent' else 'A', number(r.cosine_similarity, 6)]
            for r in pairs.itertuples()]
appendix += long_table('Tutti i valori cosine considerati: 60 confronti con il padre e 334 con altri HLG.',
                       'tab:all-cosine-values', 'lrrrr', ['Dataset', '$i$', '$j$', 'Tipo', 'Cosine'], all_rows)
(OUT / 'all_cosine_values.tex').write_text(appendix, encoding='utf-8')

# Reference results are from frozen v103 outputs at temperature 0.3, never relabeled 0.2.
ref = comparison[comparison.Esperimento == 'reference'].set_index('Dataset')
ref_metrics = metrics[metrics.config_id == 'reference'].set_index(['dataset', 'level'])
base_metrics = metrics[metrics.config_id == 'top_down'].set_index(['dataset', 'level'])
assert set(ref.index) == set(short)
trace_evidence = {}
for index, doc in enumerate(snapshot['documents']):
    run = doc['runs']['reference/repeat_001']
    cycle = run['globalGoalCycle']
    decisions = Counter(e.get('decision') or 'INCONCLUSIVE' for t in cycle['iterations'] for e in t['evaluations'].values())
    outcomes = Counter(v for t in cycle['iterations'] for v in t.get('structural_action_outcomes', {}).values())
    for level in ['HLG', 'LLG']:
        row = ref_metrics.loc[(doc['name'], level)]
        final = cycle['final_high_level_goals']['goals'] if level == 'HLG' else cycle['final_low_level_goals']['low_level_goals']
        assert int(row.generated_goal_count) == len(final)
    trace_evidence[doc['name']] = {'json_pointer': f'/documents/{index}/runs/reference~1repeat_001',
                                  'decisions': dict(decisions), 'outcomes': dict(outcomes),
                                  'iterations': len(cycle['iterations']), 'converged': run['converged']}

reference = r'''\section{Analisi dettagliata della configurazione reference}
\label{sec:reference-dataset-analysis}
% Evidence C-R01/C-R02: frozen final_metrics.csv, dataset_comparison.csv, source_snapshot.json.
La configurazione \texttt{reference}, indicata come P4, usa quattro scenari
few-shot. I risultati di questa sezione appartengono alla campagna storica
\path{ofat_fixed_thresholds_20261024_211627}, con temperatura 0,3, soglia 0,68,
policy v103 e una run per ciascuno dei dodici dataset. Non sono risultati
della nuova reference a temperatura 0,2 o degli esempi aggiornati successivamente.
Il confronto è appaiato con la baseline top-down dello stesso progetto.

La reference raggiunge F1 macro 0,53723 per gli HLG e 0,51528 per gli LLG.
Per gli HLG quattro dataset migliorano, quattro peggiorano e quattro restano
invariati; per gli LLG i conteggi sono rispettivamente cinque, tre e quattro.
Dieci run convergono, mentre Fitness Tracker e SIA 25--26 raggiungono il limite
di 25 iterazioni. La convergenza non è una misura di correttezza documentale.

'''
rows = []
for name in short:
    row = ref.loc[name]
    b_h, r_h = base_metrics.loc[(name, 'HLG')], ref_metrics.loc[(name, 'HLG')]
    b_l, r_l = base_metrics.loc[(name, 'LLG')], ref_metrics.loc[(name, 'LLG')]
    trace = trace_evidence[name]
    rows.append([short[name], number(b_h.f1), number(r_h.f1), number(row['HLG Delta F1'], signed=True),
                 number(b_l.f1), number(r_l.f1), number(row['LLG Delta F1'], signed=True),
                 str(trace['iterations']) + (r'$^{*}$' if not trace['converged'] else '')])
reference += table('Reference storica: F1 prima e dopo il feedback. I delta sono differenze assolute; l’asterisco indica mancata convergenza.',
                   'tab:reference-before-after', 'lrrrrrrr',
                   ['Dataset', '$H_0$', '$H_R$', '$\Delta H$', '$L_0$', '$L_R$', '$\Delta L$', 'Iter.'], rows)
reference += r'''\paragraph{Come vengono interpretati i cambiamenti.}
Le metriche usano un matching testuale uno-a-uno. Il numero di goal, la loro
copertura e la formulazione influenzano congiuntamente il punteggio; un aumento
di F1 non certifica ogni requisito o associazione padre--figlio. Le spiegazioni
seguenti collegano metriche e modifiche osservate, senza attribuire causalmente
l'esito al solo few-shot: non sono disponibili repliche o confronti controllati
che isolino la causa di ogni differenza.
I conteggi di riscritture e dei relativi esiti sono conteggi per ramo lungo
le iterazioni, non chiamate API indipendenti. Una singola riparazione batch
può produrre un esito applicato per più rami; i passaggi batch riportati
nel riepilogo operativo vanno quindi distinti da questi conteggi.

'''
narratives = {
    'Assegno Unico Universale - SIA Project 24 25': r'''Gli HLG passano da 10 a 11 a fronte di quattro riferimenti. La Recall HLG rimane quasi stabile (0,7760--0,7795), mentre la Precision scende da 0,3104 a 0,2835: il lieve aumento della somma del matching non compensa l'ulteriore goal nel denominatore. Gli LLG passano invece da 24 a 30 su 28 riferimenti e la Recall sale da 0,6719 a 0,7795. Le 18 iterazioni comprendono 56 decisioni REWRITE, 52 passaggi di riparazione accettati e tre rimozioni per duplicazione. Queste azioni convivono con la rigenerazione dell'intera collezione: il numero finale degli HLG non è quindi la sola baseline meno le rimozioni. L'esito è misto, con maggiore copertura LLG ma frammentazione degli HLG.''',
    'Calculator': r'''La run termina alla prima iterazione con un KEEP e mantiene un HLG e cinque LLG. Entrambi i valori F1 restano identici alla baseline. Il confronto contiene due HLG e sette LLG di riferimento: confermare il ramo esistente conserva la gerarchia, ma non recupera le capacità che non erano state estratte. Questo caso mostra una conservazione operativa della baseline, non un miglioramento dell'estrazione.''',
    'Ethical Purchasing Group - SIA Project 22 23': r'''Tutti i nove rami ricevono KEEP nella prima iterazione e restano nove HLG e 22 LLG. La ground truth contiene quattro HLG e 14 LLG. La Precision HLG di 0,3009 e quella LLG di 0,4162 non cambiano: il metodo conferma una rappresentazione più articolata dei riferimenti senza correggerne il disallineamento misurato. Poiché non vi sono riparazioni, l'assenza di variazioni è direttamente verificabile nelle tracce.''',
    'Event Organization Portal - SIA Project 21 22': r'''Due REWRITE nella prima iterazione, entrambi applicati, sono seguiti dalla conferma alla seconda. I conteggi rimangono tre HLG e dieci LLG, contro quattro e 28 riferimenti. Le critiche riguardano ambito della gestione finanziaria, inviti, richieste e responsabilità dei fornitori; alcune indicazioni sulla gestione degli ospiti non coincidono tra loro. La lieve riduzione HLG e il piccolo incremento LLG sono quindi associati a una riformulazione con numerosità invariata. La Recall LLG finale resta 0,2658: il miglioramento non recupera l'ampia parte della collezione ancora non coperta.''',
    'Fitness Tracker': r'''Dopo 25 iterazioni la run non converge. Cinquanta decisioni REWRITE producono 47 riparazioni accettate e tre rifiutate; sono registrati sette esiti inconcludenti e due rami rimossi, uno come duplicato. Gli HLG scendono da sette a cinque e gli LLG da 17 a 11. La Precision LLG cresce da 0,5548 a 0,6914, ma la Recall diminuisce da 0,7255 a 0,5851: il piccolo aumento di F1 deriva da questo compromesso, non da una maggiore copertura. Le critiche finali sulla definizione di obiettivi e scadenze sono incompatibili, come documentato nel caso qualitativo già discusso. Il lieve guadagno metrico non risolve l'instabilità del ciclo.''',
    'Genome Nexus': r'''Due KEEP arrestano il ciclo alla prima iterazione. Restano due HLG e otto LLG, contro tre e 30 riferimenti. La Precision LLG è 0,8608 ma la Recall è soltanto 0,2296: i pochi goal generati possono avere buone corrispondenze pur rappresentando una parte limitata dei riferimenti. L'invarianza non costituisce evidenza di completezza; mostra che la revisione dei rami presenti non ha attivato una nuova decomposizione.''',
    'Gestao Hospital': r'''Cinque KEEP nella prima iterazione mantengono cinque HLG e 13 LLG, contro quattro e 18 riferimenti. F1 HLG resta 0,6817 e F1 LLG 0,6199. Gli evaluator non propongono interventi e il sistema preserva una baseline già relativamente vicina ai riferimenti secondo il matching. Il risultato è stabile, senza evidenza di un guadagno dovuto al feedback.''',
    'La Reine Marlene - SIA Project 23 24': r'''È il secondo maggiore incremento LLG della reference: +0,0991. Tre riparazioni accettate in tre iterazioni mantengono tre HLG, ma portano gli LLG da dieci a 13 su 29 riferimenti. La Recall LLG aumenta da 0,2638 a 0,3558 e la Precision da 0,7649 a 0,7937. La decomposizione finale esplicita categoria, prezzo, disponibilità, immagini, descrizione e tempi di consegna delle proposte, oltre a revisione amministrativa e ricezione dei prodotti. È una spiegazione compatibile con l'aumento della copertura misurata, ma anche la baseline già conteneva una proposta generica: parte del guadagno riguarda granularità e corrispondenza testuale. Gli HLG restano tre contro dieci riferimenti e il loro F1 diminuisce leggermente; il miglioramento è localizzato alla decomposizione.''',
    'London Ambulance Service': r'''È il maggiore peggioramento LLG: $-0{,}1964$, nonostante la convergenza in 13 iterazioni e un incremento HLG di +0,0189. Diciannove REWRITE producono 18 riparazioni accettate; un ramo è rimosso come duplicato. Gli LLG raddoppiano da sei a dodici contro tre riferimenti. La Recall rimane quasi stabile (0,7262--0,7194), mentre la Precision si dimezza da 0,3631 a 0,1798: il maggiore denominatore spiega il forte calo di F1 nel matching uno-a-uno. Gli output includono traffico in tempo reale e previsione dell'arrivo, dettagli non richiesti esplicitamente dal testo di partenza. Più dettaglio e conferma degli evaluator non equivalgono a un'estrazione migliore.''',
    'Planning Poker': r'''È il maggiore calo HLG della reference: $-0{,}0366$. Gli HLG restano tre contro cinque riferimenti, quindi il peggioramento non dipende dal loro numero ma dalla somma delle corrispondenze, che diminuisce con la riformulazione. Dopo 13 riparazioni accettate e una rimozione per duplicazione, la run converge in nove iterazioni. Gli LLG passano da nove a otto su 33 riferimenti; la Precision sale da 0,7294 a 0,7715 ma la Recall scende da 0,1989 a 0,1870, portando F1 a 0,3011. Gli output si concentrano su stime anonime, visualizzazione del consenso e riordinamento delle storie. Questa concentrazione non recupera la parte più ampia delle funzionalità di riferimento.''',
    'Recycling': r'''È il maggiore incremento HLG, +0,1275, ma uno dei peggioramenti LLG più ampi, $-0{,}0790$. Gli HLG passano da sette a cinque, lo stesso numero dei riferimenti. La Recall HLG cambia pochissimo (0,7573--0,7587), mentre la Precision sale da 0,5409 a 0,7587: il guadagno è compatibile soprattutto con la riduzione di ridondanza o frammentazione nel matching. Gli LLG scendono da 15 a 12 su 25 riferimenti; la Precision resta quasi invariata, ma la Recall passa da 0,4725 a 0,3786. La traccia contiene la rimozione del ramo relativo a notifiche degli eventi e navigazione, undici riparazioni accettate e una decomposizione finale senza un goal esplicito per la mappa. Le notifiche degli eventi ricompaiono invece in un LLG finale: non sono interamente assenti. Compaiono anche AES-256 e TLS 1.3, dettagli non specificati nella documentazione. Il vantaggio agli HLG non dimostra la preservazione completa delle funzionalità dei figli.''',
    'SIA Project 25 26': r'''È il maggiore miglioramento LLG: +0,2309, accompagnato da +0,0285 negli HLG. I figli passano da 14 a 28 su 31 riferimenti. La Recall LLG aumenta da 0,3508 a 0,6797, con una riduzione più contenuta della Precision da 0,7767 a 0,7525. La gerarchia finale esplicita posizione, titolo, categoria, fotografie e anonimato del report, motivazione dei rifiuti, configurazione degli assegnamenti e statistiche pubbliche e private. Questa maggiore copertura è osservabile nei testi e coerente con il guadagno di Recall. Tuttavia, sono necessarie 25 iterazioni senza convergenza: 78 decisioni REWRITE, 64 riparazioni accettate, 14 rifiutate e sette esiti inconcludenti. Il miglioramento non è una prova di stabilità del meccanismo.''',
}

for name in short:
    row = ref.loc[name]
    reference += '\\subsection{' + short[name] + '}\n'
    reference += '% Evidence C-R02: source_snapshot.json ' + trace_evidence[name]['json_pointer'] + '\n'
    reference += 'Rispetto al top-down, F1 HLG cambia di $' + number(row['HLG Delta F1'], signed=True) + '$ e F1 LLG di $' + number(row['LLG Delta F1'], signed=True) + '$. '
    reference += narratives[name] + '\n\n'
reference += r'''\paragraph{Confronto tra gli estremi.}
SIA 25--26 e London mostrano che il numero di LLG non ha un effetto uniforme:
nel primo caso cresce la copertura di una collezione di 31 riferimenti;
nel secondo aumentano goal e dettagli rispetto a tre riferimenti senza un
incremento della somma del matching. Recycling mostra inoltre che una riduzione
dei padri può migliorare F1 HLG mentre si perde copertura ai figli.
Queste osservazioni spiegano i punteggi attraverso conteggi e tracce, ma non
isolano la qualità semantica di ciascuna modifica o l'effetto causale del few-shot.
'''
(OUT / 'reference_dataset_analysis.tex').write_text(reference, encoding='utf-8')

files = [TUNING / n for n in ['parent_review.json', 'cosine_summary.json', 'cosine_pairs.csv', 'cosine_by_dataset.csv']]
files += [RESULTS / n for n in ['source_snapshot.json', 'final_metrics.csv', 'dataset_comparison.csv', 'operational_summary.csv', 'case_evidence.json']]
evidence = {'created_for': 'main.tex additions, 2026-10-26',
            'sources': {str(p.relative_to(ROOT)): digest(p) for p in files},
            'claims': {'C-T01': 'Tuning aggregates verified against 60 parent pairs and 12 dataset means',
                       'C-T02': 'All 394 unique pair rows and descriptive threshold counts; no equivalence truth labels',
                       'C-R01': 'Reference v103 metrics at temperature 0.3; not the new 0.2 protocol',
                       'C-R02': trace_evidence},
            'machine_consistency_checks': 'passed', 'human_author_verification': 'pending',
            'interpretation': 'Descriptive, post hoc trace analysis. No fabricated manual equivalence annotations or validated optimum.'}
(OUT / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

main = ROOT / 'main.tex'
text = main.read_text(encoding='utf-8')
if r'\usepackage{longtable}' not in text:
    text = text.replace(r'\usepackage{tabularx}', '\\usepackage{tabularx}\n\\usepackage{longtable}')
def insert_or_update(start_marker, next_marker, body):
    global text
    if start_marker in text:
        start = text.index(start_marker)
        end = text.index(next_marker, start)
        text = text[:start] + body.rstrip() + '\n\n' + text[end:]
    else:
        text = text.replace(next_marker, body.rstrip() + '\n\n' + next_marker, 1)


insert_or_update(r'\subsection{Analisi esplorativa della soglia su tutti i dataset}',
                 r'\section{Configurazioni degli evaluator}', calibration)
insert_or_update(r'\section{Analisi dettagliata della configurazione reference}',
                 r'\section{Convergenza, decisioni e costo iterativo}', reference)
insert_or_update(r"\chapter{Valori completi dell'analisi cosine}",
                 r'\chapter{Prompt e dettagli implementativi}', appendix)
text = text.replace('The experimental study varies only', 'The main configuration study varies only')
exploratory_sentence = ' A separate retrospective exploratory analysis reports all observed cosine values and their sensitivity to candidate thresholds; it does not establish a statistically optimal threshold.'
text = text.replace(exploratory_sentence, '')
text = text.replace('Cosine similarity remains enabled at threshold 0.68 for all datasets and configurations.',
                    'Cosine similarity remains enabled at threshold 0.68 for all datasets and configurations.' + exploratory_sentence)
text = text.replace("scelta sulla base delle prove preliminari effettuate e applicata", "adottata come valore operativo e applicata")
text = text.replace('la mappa e le notifiche degli eventi, citate nella\ndescrizione.', 'la mappa citata nella descrizione. Le notifiche degli eventi sono invece\npresenti in un LLG finale e non possono essere descritte come interamente assenti.')
text = text.replace('non sono espliciti la mappa citata', 'non è esplicita la mappa citata')
text = text.replace('a 0,7586, mentre', 'a 0,7587, mentre')
# Remove redundant pair metadata from the methodological description.
text = text.replace('soltanto le coppie candidate: identificativo e padre del ramo esaminato,\nHLG$\'$ di partenza, identificativo dell\'altro HLG, suo nome e descrizione,\nattore comune ed etichetta \\texttt{POTENTIAL\\_DUPLICATE}.',
                    "soltanto le coppie candidate: identificativo del ramo e HLG$'$ di partenza,\nidentificativo dell'altro HLG e suo nome e descrizione. Il padre originale\ne l'attore restano nelle specifiche del ramo e non sono ripetuti nella coppia.")
from src.bottom_up import global_goal_evaluator as current
for label, body in [('lst:evaluator-decision-instructions-full', current.GLOBAL_REVIEW_INSTRUCTIONS),
                    ('lst:evaluator-prompt-assembly', inspect.getsource(current._critic1_prompt))]:
    a = text.index('label={' + label + '}')
    a = text.index('\n', a) + 1
    z = text.index('\\end{lstlisting}', a)
    text = text[:a] + body.rstrip() + '\n' + text[z:]
text = text.replace('versione corrente v111.', 'versione corrente v112.')
main.write_text(text, encoding='utf-8')
print('Added tuning, all 394 scores, reference analysis for 12 datasets, and evidence hashes.')
