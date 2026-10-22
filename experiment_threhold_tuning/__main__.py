"""Frozen reconstruction/ground-truth cosine experiment, separate from feedback.

Run from the repository root: python -m experiment_threhold_tuning --help
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
PAIR_TYPES = ('prime_ground_truth', 'prime_parent', 'prime_prime_duplicate')


def checksum(raw):
    return hashlib.sha256(raw).hexdigest()


def child_signature(collection):
    return Counter((g['name'], g['description'], g['high_level_associated']['name'],
                    g['high_level_associated']['description']) for g in collection['low_level_goals'])


def saved_reconstructions(trace, baseline, groups):
    """Reuse only initial reconstructions with the same input children and groups."""
    initial = (trace.get('input_hierarchy') or {}).get('low_level_goals')
    if trace.get('iteration') != 1 or initial is None:
        raise ValueError('Saved reconstruction must identify its iteration-1 input hierarchy')
    if child_signature(initial) != child_signature(baseline['structuredLowLevelGoals']):
        raise ValueError('Saved reconstruction inputs differ from the current baseline')
    results, consumed = {}, set()
    for group in groups:
        names = Counter(child.name for child in group.low_level_goals)
        matches = [(bid, value) for bid, value in trace.get('reconstructions', {}).items()
                   if Counter(value.get('source_low_level_goal_ids', [])) == names]
        if len(matches) != 1 or matches[0][0] in consumed:
            raise ValueError('Saved reconstructions do not map unambiguously to the current LLG groups')
        bid, value = matches[0]
        text = value.get('reconstructed_high_level_goal')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Missing reconstructed HLG text')
        consumed.add(bid)
        results[group.branch_id] = text
    return results


def dataset_splits(names, validation_datasets=None, seed=42):
    names = sorted(names)
    if validation_datasets is None:
        shuffled = names.copy()
        random.Random(seed).shuffle(shuffled)
        validation = set(shuffled[:max(1, len(names)//4)]) if len(names) > 1 else set()
    else:
        validation = set(validation_datasets)
        if not validation <= set(names):
            raise ValueError(f'Unavailable validation datasets: {sorted(validation - set(names))}')
    if names and set(names) == validation:
        raise ValueError('At least one dataset must remain for calibration')
    return {name: 'validation' if name in validation else 'calibration' for name in names}


def build_pairs(dataset, groups, primes, references, split, encode):
    """Use the exact embedding and cosine functions used in bottom-up feedback."""
    from src.bottom_up.goal_similarity import _cosine
    rows = []

    def add(kind, left_id, right_id, left, right):
        score = _cosine(encode(left), encode(right))
        identity = json.dumps([dataset, kind, left_id, right_id, left, right], ensure_ascii=False)
        rows.append(dict(pair_id=checksum(identity.encode())[:24], dataset=dataset, split=split,
            pair_type=kind, left_id=left_id, right_id=right_id,
            left_text=left, right_text=right, score=score))

    for group in groups:
        prime = primes[group.branch_id]
        for index, reference in enumerate(references):
            add('prime_ground_truth', group.branch_id, f'gt_{index:03d}', prime, reference)
        parent = group.high_level_goal
        add('prime_parent', group.branch_id, group.branch_id, prime,
            f'{parent.name}\n{parent.description}')
    for index, left in enumerate(groups):
        for right in groups[index+1:]:
            # Match the duplicate population used by the current feedback cycle.
            actor_key = lambda group: ' '.join(group.high_level_goal.actor.name.casefold().split())
            if actor_key(left) == actor_key(right):
                add('prime_prime_duplicate', left.branch_id, right.branch_id,
                    primes[left.branch_id], primes[right.branch_id])
    return rows


def prepare(*, root=ROOT, source='saved', datasets=None, validation_datasets=None,
            seed=42, run_id=None, encode=None, reconstruct=None, references=None):
    from src.data_model import LowLevelGoals, group_low_level_goals_by_hlg
    from src.bottom_up import goal_similarity as similarity
    if references is None:
        from notebook.groundtruth import GROUNDTRUTH_BY_NAME
        references = GROUNDTRUTH_BY_NAME
    root = Path(root).resolve()
    run_id = run_id or datetime.now(ZoneInfo('Europe/Rome')).strftime('run_%Y%m%d_%H%M%S_%f')
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('run_id must contain only letters, digits, underscore or hyphen')
    directory = root/'experiment_threhold_tuning'/'outputs'/run_id
    if directory.exists():
        raise FileExistsError(f'Run gia esistente: {directory}. Scegli un altro --run-id.')
    baseline_paths = sorted((root/'output'/'top_down').glob('*_top_down.json'))
    available_names = {json.loads(p.read_bytes())['name'] for p in baseline_paths}
    if datasets and not set(datasets) <= available_names:
        raise ValueError(f'Unknown datasets: {sorted(set(datasets) - available_names)}')
    candidates, coverage = [], []
    for path in baseline_paths:
        raw = path.read_bytes()
        baseline = json.loads(raw)
        name = baseline['name']
        if datasets and name not in datasets:
            continue
        print(f'[prepare] {name}', flush=True)
        try:
            if name not in references or not references[name]['highLevelGoals']:
                raise ValueError('Missing HLG ground truth')
            groups = group_low_level_goals_by_hlg(LowLevelGoals.model_validate(baseline['structuredLowLevelGoals']))
            if not groups:
                raise ValueError('No nonempty LLG groups to reconstruct')
            provenance = dict(baseline_file=str(path), baseline_sha256=checksum(raw), source=source)
            if source == 'saved':
                from src.bottom_up.result_storage import dataset_filename
                compact = root/'output'/'top_down_bottom_up'/'datasets'/dataset_filename(name)
                if compact.exists():
                    trace_paths = [compact]
                    trace_raw = compact.read_bytes()
                    document = json.loads(trace_raw)
                    traces = [t for r in document['runs'].values()
                              for t in r.get('globalGoalCycle', {}).get('iterations', [])
                              if t.get('iteration') == 1 and t.get('checkpoint_complete', True)]
                    if len(traces) != 1:
                        raise ValueError('Expected one completed initial reconstruction in the dataset JSON')
                    trace = traces[0]
                else:
                    trace_paths = list((root/'output'/'bottom_up_iterations'/name).glob('*iteration_001.json'))
                    if len(trace_paths) != 1:
                        raise ValueError('Expected one saved iteration-1 trace; use --source fresh to reconstruct')
                    trace_raw = trace_paths[0].read_bytes()
                    trace = json.loads(trace_raw)
                primes = saved_reconstructions(trace, baseline, groups)
                provenance.update(reconstruction_file=str(trace_paths[0]), reconstruction_sha256=checksum(trace_raw),
                    generation_provenance='Historical saved text; generator/prompt uniformity is not guaranteed')
            elif source == 'fresh':
                from src.bottom_up.goal_reconstructor import reconstruct_all_branches
                from src.llm_clients import QWEN_GENERATOR_MODEL
                values = (reconstruct or reconstruct_all_branches)(groups)
                primes = {key: value.reconstructed_high_level_goal for key, value in values.items()}
                provenance.update(generator_model=QWEN_GENERATOR_MODEL, generation_provenance='Fresh reconstruction; no ground truth in generator input')
            else:
                raise ValueError('source must be saved or fresh')
            candidates.append((name, groups, primes, baseline, provenance))
            coverage.append(dict(dataset=name, status='included', reason='', group_count=len(groups)))
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            coverage.append(dict(dataset=name, status='excluded', reason=f'{type(exc).__name__}: {exc}', group_count=0))
    if not candidates:
        reasons = '\n'.join(f"- {row['dataset']}: {row['reason']}" for row in coverage)
        raise ValueError('Nessun dataset disponibile. Verifica le baseline in output/top_down.\n'
                         'Per nuove ricostruzioni usa prepare --source fresh.\n' + reasons)
    splits = dataset_splits([x[0] for x in candidates], validation_datasets, seed)
    code = [Path(__file__), Path(__file__).with_name('workbook.py'), ROOT/'src/bottom_up/goal_similarity.py',
            ROOT/'src/bottom_up/goal_reconstructor.py']
    manifest = dict(schema_version=1, run_id=run_id, source=source, seed=seed, splits=splits,
        status='preparing', created_at=datetime.now(ZoneInfo('Europe/Rome')).isoformat(),
        embedding_configuration=similarity.similarity_configuration(),
        pair_types=list(PAIR_TYPES), code_sha256={str(p.relative_to(ROOT)): checksum(p.read_bytes()) for p in code},
        excluded=[row for row in coverage if row['status'] == 'excluded'],
        ground_truth_usage='Post-hoc comparisons only; never passed to reconstruction or feedback',
        threshold_policy='No labels inferred from ground truth membership, score or nearest match; no live threshold changes')
    encoder = encode or similarity._encode_text
    pairs, snapshots = [], []
    for name, groups, primes, baseline, provenance in candidates:
        refs = references[name]['highLevelGoals']
        frozen = dict(dataset=name, split=splits[name], provenance=provenance,
            structuredLowLevelGoals=baseline['structuredLowLevelGoals'],
            structuredHighLevelGoals=baseline['structuredHighLevelGoals'], ground_truth_hlg=refs,
            reconstructions=[dict(group_id=g.branch_id, parent=g.high_level_goal.model_dump(mode='json'),
                                  llgs=[c.model_dump(mode='json') for c in g.low_level_goals], hlg_prime=primes[g.branch_id])
                             for g in groups])
        snapshots.append(frozen)
        print(f'[cosine] {name}: {len(groups)} reconstructed HLG x {len(refs)} references', flush=True)
        current = build_pairs(name, groups, primes, refs, splits[name], encoder)
        pairs.extend(current)
    manifest.update(status='awaiting_annotations', dataset_count=len(candidates), pair_count=len(pairs),
        resolved_embedding_revision=similarity._resolved_revision() if encode is None else 'injected_test_encoder')
    from experiment_threhold_tuning.workbook import write_workbook, FILENAME
    write_workbook(directory/FILENAME, manifest, coverage, snapshots, pairs)
    print(f'Saved {len(pairs)} pairs in {directory/FILENAME}; human annotations and confirmation required.', flush=True)
    return directory


def workbook_path(directory=None):
    """Find the only workbook, or require an explicit choice when ambiguous."""
    from experiment_threhold_tuning.workbook import FILENAME, consolidate
    if directory is not None:
        path = Path(directory).resolve()
        return consolidate(path) if path.is_dir() else path
    paths = sorted((ROOT/'experiment_threhold_tuning'/'outputs').glob(f'*/{FILENAME}'))
    if not paths:
        raise ValueError('Nessun workbook trovato. Esegui prima: python -m experiment_threhold_tuning prepare')
    if len(paths) > 1:
        choices = '\n'.join(str(path) for path in paths)
        raise ValueError('Sono presenti piu workbook: specifica tune "percorso.xlsx".\n' + choices)
    return paths[0]


def tune(directory=None):
    """Refresh the Excel proposal; final threshold choices remain manual."""
    from experiment_threhold_tuning.workbook import tune_workbook
    path = workbook_path(directory)
    print(f'Workbook: {path}', flush=True)
    report = tune_workbook(path)
    for kind, result in report.items():
        print(f"{kind}: {result['labeled']}/{result['total']} annotazioni | "
              f"{result['status']} | proposta={result.get('proposed_threshold')} | "
              f"confermata={result.get('confirmed_threshold')}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(prog='python -m experiment_threhold_tuning',
                                     description='Prepara un Excel di confronti e aggiorna il tuning dopo le annotazioni.')
    commands = parser.add_subparsers(dest='command', required=True)
    preparation = commands.add_parser('prepare', help='Freeze HLG reconstructions and compute cosine pairs')
    preparation.add_argument('--source', choices=['saved', 'fresh'], default='saved')
    preparation.add_argument('--datasets', nargs='+')
    preparation.add_argument('--validation-datasets', nargs='+')
    preparation.add_argument('--seed', type=int, default=42)
    preparation.add_argument('--run-id')
    migration = commands.add_parser('consolidate', help='Export an archived run to one Excel without recomputing cosine')
    migration.add_argument('directory', type=Path)
    tuning = commands.add_parser('tune', help='Update Excel threshold proposals from manual annotations')
    tuning.add_argument('directory', type=Path, nargs='?', help='Excel o cartella; omettibile se esiste un solo workbook')
    args = vars(parser.parse_args())
    command = args.pop('command')
    try:
        if command == 'prepare':
            prepare(**args)
        elif command == 'consolidate':
            from experiment_threhold_tuning.workbook import consolidate
            print(consolidate(**args))
        else:
            tune(**args)
    except PermissionError:
        parser.exit(1, 'Chiudi il file Excel e riprova. Verifica anche i permessi della cartella.\n')
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
