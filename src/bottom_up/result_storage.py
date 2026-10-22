"""One atomically updated JSON per dataset, plus one batch summary.

Configurations/repetitions are entries in a dataset document, not extra files.
This module performs no model calls and does not decide goal actions.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def now():
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode('utf-8')


def payload_hash(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dataset_filename(name):
    slug = re.sub(r'[^\w.-]+', '_', name).strip('._')
    if not slug:
        raise ValueError('Dataset name cannot be empty')
    # Hash prevents collisions while keeping filenames readable.
    return f'{slug[:100]}_{hashlib.sha256(name.encode()).hexdigest()[:8]}.json'


def _plain(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json', exclude_defaults=True)
    return deepcopy(value)


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()
                if k not in {'raw_response', 'attempt_history', 'stage_logs', 'output_hierarchy'}}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _difference(before, after):
    remaining = Counter(json.dumps(g, sort_keys=True, ensure_ascii=False) for g in before)
    added = []
    for goal in after:
        key = json.dumps(goal, sort_keys=True, ensure_ascii=False)
        if remaining[key]:
            remaining[key] -= 1
        else:
            added.append(goal)
    removed = [json.loads(key) for key, count in remaining.items() for _ in range(count)]
    return added, removed


def compact_cycle(value):
    """Retain decisions and actual changes; omit raw prompts/retry transcripts."""
    cycle = _clean(_plain(value))
    for trace in cycle.get('iterations', []):
        parents = {}
        for child in (trace.get('input_hierarchy') or {}).get('low_level_goals', {}).get('low_level_goals', []):
            parent = child.get('high_level_associated')
            if parent:
                text = json.dumps((parent['name'], parent['description']), ensure_ascii=False, separators=(',', ':'))
                key = 'hlg_' + hashlib.sha256(text.encode()).hexdigest()[:16] + '_001'
                parents[key] = parent
        decisions = []
        for branch_id, evaluation in trace.get('evaluations', {}).items():
            votes = evaluation.get('critic_votes', {}).get('documentary', [])
            decisions.append(dict(branch_id=branch_id, model_decisions=votes,
                                  original_hlg=parents.get(branch_id),
                                  reconstructed_hlg=trace.get('reconstructions', {}).get(branch_id),
                                  final_decision=evaluation['final_decision'],
                                  rationale=evaluation.get('rationale'),
                                  feedback=evaluation.get('feedback'),
                                  applied_change=trace.get('structural_action_outcomes', {}).get(branch_id)))
        trace['decision_summary'] = decisions
        if trace.get('checkpoint_complete', True):
            initial = trace.get('input_hierarchy') or {}
            changes = {}
            for prefix, field, collection in (('hlg', 'high_level_goals', 'goals'),
                                               ('llg', 'low_level_goals', 'low_level_goals')):
                before = initial.get(field, {}).get(collection, [])
                after = trace.get(field, {}).get(collection, [])
                added, removed = _difference(before, after)
                changes[f'{prefix}_added'], changes[f'{prefix}_removed'] = added, removed
            trace['changes'] = changes
    return cycle


class DatasetResults:
    def __init__(self, path, baseline, *, reset=False):
        self.path = Path(path)
        if self.path.exists() and not reset:
            self.data = read_json(self.path)
            if self.data.get('baseline_sha256') != payload_hash(baseline):
                raise ValueError('Dataset baseline changed; use a new suite')
        else:
            self.data = dict(schema_version=2, name=baseline['name'], baseline=baseline,
                             baseline_sha256=payload_hash(baseline), runs={})

    def save(self):
        self.data['updated_at'] = now()
        atomic_write(self.path, self.data)

    def start(self, run_id, *, configuration=None, repetition=1):
        previous = self.data['runs'].get(run_id, {})
        self.data['runs'][run_id] = dict(status='running', started_at=now(),
            attempt=previous.get('attempt', 0) + 1, experiment=configuration,
            repetition=repetition, converged=None, stop_reason=None,
            globalGoalCycle={'iterations': [], 'completed_iterations': 0}, final_decisions=[])
        self.save()

    def update(self, run_id, cycle):
        record = self.data['runs'][run_id]
        record['globalGoalCycle'] = compact_cycle(cycle)
        iterations = record['globalGoalCycle'].get('iterations', [])
        record['completed_iterations'] = sum(t.get('checkpoint_complete', True) for t in iterations)
        if iterations:
            last = iterations[-1]
            record['currentHighLevelGoals'] = last['high_level_goals']
            record['currentLowLevelGoals'] = last['low_level_goals']
            record['final_decisions'] = last.get('decision_summary', [])
        self.save()

    def complete(self, run_id, result):
        self.update(run_id, result)
        record = self.data['runs'][run_id]
        cycle = record['globalGoalCycle']
        record.update(status='completed', finished_at=now(), converged=cycle['converged'],
                      stop_reason=cycle['stop_reason'],
                      finalHighLevelGoals=cycle['final_high_level_goals'],
                      finalLowLevelGoals=cycle['final_low_level_goals'])
        self.save()

    def fail(self, run_id, error, *, interrupted=False):
        record = self.data['runs'][run_id]
        record.update(status='interrupted' if interrupted else 'failed', error=str(error),
                      finished_at=now(), converged=None, stop_reason='EXECUTION_ERROR')
        self.save()


def result_payload(document, run):
    """Common adapter for evaluation of compact and legacy results."""
    return {**document['baseline'], 'baselineSha256': document['baseline_sha256'],
            **run}


def write_summary(directory, dataset_files, *, planned_runs=None):
    documents = [read_json(path) for path in dataset_files if Path(path).exists()]
    runs = [run for document in documents for run in document['runs'].values()]
    counts = Counter(run['status'] for run in runs)
    expected = planned_runs if planned_runs is not None else len(runs)
    terminal = counts['completed'] + counts['failed'] + counts['interrupted']
    complete = len(runs) == expected and terminal == expected
    summary = dict(schema_version=2, updated_at=now(), planned_runs=expected,
        status_counts=dict(counts), all_runs_completed=counts['completed'] == expected and complete,
        system_status=('COMPLETED_WITH_ERRORS' if complete and terminal != counts['completed'] else
                       'COMPLETED' if complete else 'IN_PROGRESS'),
        converged=(all(run.get('converged') is True for run in runs) if complete and runs else None),
        datasets=documents)
    atomic_write(Path(directory)/'results_summary.json', summary)
    return summary
