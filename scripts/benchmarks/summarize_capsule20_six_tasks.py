"""Audit Capsule traces, task completion, and strict execution success."""
import argparse
import json
from pathlib import Path

from summarize_singleturn_six_tasks import summarize

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
args = parser.parse_args()
result = summarize(args.root)
for task, data in result['tasks'].items():
    data['nonzero_sandbox_rc'] = data.pop('program_errors')
    data['task_completed_seeds'] = [r['trial'] for r in data['records']
                                    if r['run_outcome'] == 'finished' and r['task_completed']]
    directory = args.root / 'Qwen2.5-Coder-7B-Instruct' / task
    lengths = {}
    for path in sorted(directory.glob('capsule_trace_trial_*.json')):
        rows = json.loads(path.read_text())
        steps = [r['step_id'] for r in rows if 'step_id' in r]
        assert all(1 <= step <= 20 for step in steps), path
        lengths[path.stem] = len(steps)
    data['trace_count'] = len(lengths)
    data['step_counts'] = lengths
    data['max_observed_steps'] = max(lengths.values(), default=0)
(args.root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
for data in result['tasks'].values():
    del data['records']
    del data['step_counts']
print(json.dumps(result, indent=2))
