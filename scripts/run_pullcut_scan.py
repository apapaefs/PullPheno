"""Execute a configured study while keeping source runs immutable."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import HwSimPythonAnalysis as analysis
from scripts.pullcut_study_config import (
    analysis_label, comparison_arguments, comparison_key, comparison_label,
    event_arguments, validate_inputs,
)

CODE = Path(__file__).resolve().parents[1]


def write_json(path, value):
    analysis.atomic_write_text(path, json.dumps(value, indent=2) + '\n')


def validate_threshold_pair(looser, tighter):
    """Changing the pull mask cannot change reconstruction or event selection."""
    old = {result.spec.name: result for result in looser}
    for result in tighter:
        baseline = old[result.spec.name]
        assert result.processed_entries == baseline.processed_entries
        np.testing.assert_allclose(result.generated_sumw, baseline.generated_sumw, rtol=1e-12)
        for step, stat in result.cutflow.items():
            previous = baseline.cutflow[step]
            assert stat.raw_count == previous.raw_count
            np.testing.assert_allclose([stat.sumw, stat.sumw2], [previous.sumw, previous.sumw2],
                                       rtol=1e-9, atol=1e-7)
        for key, histogram in result.histograms.items():
            if key not in analysis.PULL_OBSERVABLE_KEYS:
                np.testing.assert_allclose(histogram.sumw, baseline.histograms[key].sumw, rtol=1e-9, atol=1e-7)
                np.testing.assert_allclose(histogram.sumw2, baseline.histograms[key].sumw2, rtol=1e-9, atol=1e-7)
        accepted = sum(k * stat.raw_count for k, stat in result.pull_pass_counts.items())
        previous = sum(k * stat.raw_count for k, stat in baseline.pull_pass_counts.items())
        assert accepted <= previous
        # These campaigns have positive event weights.
        assert result.pull_total_sumw <= baseline.pull_total_sumw * (1 + 1e-10)


class Study:
    def __init__(self, args, inputs):
        self.args = args
        self.inputs = validate_inputs(inputs)
        self.batch = args.batch.resolve()
        self.output = args.output_root.resolve()
        self.path = self.batch / 'status.json'
        self.lock = threading.Lock()
        code_hashes = {str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for pattern in ('*.py', 'scripts/*.py', 'studies/*.json')
                       for p in CODE.glob(pattern)}
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'format_version': 2, 'status': 'new', 'jobs': {}, 'comparisons': {}, 'sources': {},
            'inputs': inputs, 'analysis_version': analysis.ANALYSIS_VERSION,
            'study_tag': inputs['study_tag'], 'particle_pt_min_gev': inputs.get('particle_pt_min_gev'),
            'code_sha256': code_hashes, 'git': analysis.git_provenance(CODE),
        }
        if self.state['inputs'] != inputs or self.state['code_sha256'] != code_hashes:
            raise RuntimeError('Batch inputs or frozen code changed; use a new batch')

    def update(self, label=None, **fields):
        with self.lock:
            target = self.state['jobs'].setdefault(label, {}) if label else self.state
            target.update(fields)
            write_json(self.path, self.state)
            print(label or 'batch', json.dumps(fields), flush=True)

    def invoke(self, label, arguments=None, reused=None):
        previous = self.state['jobs'].get(label, {})
        if previous.get('phase') == 'complete':
            result = Path(previous['run'])
            assert json.loads((result / 'run.json').read_text())['status'] == 'complete'
            return result
        if previous:
            raise RuntimeError(f'{label} has recoverable state; inspect its log/checkpoints before resuming')
        if reused:
            result = self.output / 'runs' / reused
            assert json.loads((result / 'run.json').read_text())['status'] == 'complete'
            self.update(label, phase='complete', run=str(result), reused=True)
            return result
        command = [sys.executable, str(CODE / 'HwSimPythonAnalysis.py'), *arguments,
                   '--output-root', str(self.output), '--run-name', label]
        write_json(self.batch / f'{label}.command.json', command)
        self.update(label, phase='running', command=command, reused=False)
        try:
            log_path = self.batch / f'{label}.log'
            with log_path.open('x') as log:
                subprocess.run(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = Path(log_path.read_text().strip().splitlines()[-1])
            metadata = json.loads((result / 'run.json').read_text())
            assert metadata['status'] == 'complete'
            analysis.validate_html_links(result / 'index.html')
            self.update(label, phase='complete', run=str(result))
            return result
        except Exception as error:
            self.update(label, phase='failed', error=str(error))
            raise

    def event_run(self, name, profile, selection, smoke=False):
        label = analysis_label(self.inputs, name, profile, selection, smoke)
        reused = (None if smoke else self.inputs.get('reuse_sources', {}).get(profile, {}).get(selection, {}).get(name))
        result = self.invoke(label, event_arguments(self.inputs, name, profile, selection, smoke), reused)
        config, samples, metadata = analysis.load_completed_run(result)
        assert metadata['analysis_version'] == analysis.ANALYSIS_VERSION
        assert config.cut_profile == profile and config.jet_acceptance_variable == 'pseudorapidity'
        assert config.pull_magnitude_min == self.inputs['selections'][selection]
        assert config.particle_pt_min_gev == self.inputs.get('particle_pt_min_gev')
        assert config.scenario.identifier == self.inputs['scenarios'][name]['identifier']
        assert len(samples) == 5 and bool(metadata['partial']) == smoke
        assert all(r.processed_entries == (min(self.inputs['smoke_events'], r.total_entries) if smoke else r.total_entries)
                   for r in samples)
        assert all(r.invalid_events == 0 for r in samples)
        analysis.validate_results(samples, smoke)
        if not smoke:
            threshold = self.inputs['selections'][selection]
            previous = self.state['sources'].get(profile, {})
            candidates = [(self.inputs['selections'][s] or 0., paths[name]) for s, paths in previous.items()
                          if name in paths and threshold is not None and (self.inputs['selections'][s] or 0.) < threshold]
            if candidates:
                _, old_path = max(candidates)
                _, old_samples, _ = analysis.load_completed_run(Path(old_path))
                validate_threshold_pair(old_samples, samples)
        self.update(label, validation={r.spec.name: {
            'processed_entries': r.processed_entries, 'generated_sumw': r.generated_sumw,
            'invalid_events': r.invalid_events, **analysis.pull_selection_summary(r),
        } for r in samples})
        return result

    def comparison(self, profile, selection, group, runs, smoke=False):
        label = comparison_label(self.inputs, profile, selection, group, smoke)
        reused = (None if smoke else self.inputs.get('reuse_comparisons', {}).get(profile, {}).get(selection, {}).get(group))
        result = self.invoke(label, comparison_arguments(self.inputs, group, runs), reused)
        metadata = json.loads((result / 'run.json').read_text())
        config = metadata['configuration']
        assert config['cut_profile'] == profile
        assert config['pull_magnitude_min'] == self.inputs['selections'][selection]
        assert config['particle_pt_min_gev'] == self.inputs.get('particle_pt_min_gev')
        assert config['jet_acceptance_variable'] == 'pseudorapidity'
        expected = [runs[name].name for name in self.inputs['comparison_sets'][group]]
        assert [source['run_id'] for source in metadata['source_runs']] == expected
        key = comparison_key(profile, selection, group)
        if smoke:
            key = 'smoke/' + key
        with self.lock:
            self.state['comparisons'][key] = str(result)
        self.update(label, validation={'plot_count': len(json.loads((result / 'summaries/plots.json').read_text()))})
        return result

    def group(self, profile, selection, smoke=False):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.inputs['concurrent_analyses']) as pool:
            futures = {name: pool.submit(self.event_run, name, profile, selection, smoke)
                       for name in self.inputs['scenarios']}
            runs = {name: future.result() for name, future in futures.items()}
        if not smoke:
            self.state['sources'].setdefault(profile, {})[selection] = {name: str(run) for name, run in runs.items()}
            self.update()
        # Comparisons are inexpensive and use independent output directories.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(self.comparison, profile, selection, group, runs, smoke)
                       for group in self.inputs['comparison_sets']]
            for future in futures:
                future.result()

    def execute(self):
        if self.args.phase == 'smoke':
            assert self.state['status'] in ('new', 'smoke')
            self.update(status='smoke')
            for selection in self.inputs['smoke_selections']:
                self.group(self.inputs['smoke_profile'], selection, smoke=True)
            self.update(status='smoke-complete')
        else:
            assert self.state['status'] in ('smoke-complete', 'production')
            self.update(status='production')
            selections = sorted(self.inputs['selections'], key=lambda s: self.inputs['selections'][s] or 0.)
            for profile in self.inputs['profiles']:
                for selection in selections:
                    self.group(profile, selection)
            self.update(status='complete')


def run(args, inputs):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    args.batch.mkdir(parents=True, exist_ok=True)
    with (args.batch / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study = Study(args, inputs)
        try:
            study.execute()
        except Exception as error:
            study.update(status='failed', error=str(error))
            raise
