#!/usr/bin/env python3
"""Separate campaign-22 smoke, ROOT analysis and saved-moment comparisons."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from scripts.pullcut_study_config import (  # noqa: E402
    analysis_label, comparison_arguments, comparison_key, comparison_label,
    event_arguments, validate_inputs,
)


def write_json(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def validate_study(inputs):
    validate_inputs(inputs)
    if inputs['selections'] != {'uncut': None} or inputs.get('reuse_sources') or inputs.get('reuse_comparisons'):
        raise ValueError('Asymmetries require nine fresh uncut ROOT analyses; old runs lack the new moments')
    if (len(inputs['scenarios']), len(inputs['profiles']), len(inputs['comparison_sets'])) != (3, 3, 3):
        raise ValueError('Expected three scenarios, acceptances and comparison groups')
    if inputs['smoke_selections'] != ['uncut'] or inputs['particle_pt_min_gev'] != .1:
        raise ValueError('Smoke must be uncut and particle pT threshold must be 0.1 GeV')
    if inputs['concurrent_analyses'] > 2 or inputs['workers'] > 190:
        raise ValueError('At most two analyses with 190 event workers each are allowed')
    return inputs


def assert_completed(directory):
    metadata = json.loads((Path(directory)/'run.json').read_text())
    if metadata['status'] != 'complete':
        raise RuntimeError(f'Run is not complete: {directory}')
    return metadata


class Study:
    def __init__(self, args, inputs, analysis_module=None):
        if analysis_module is None:
            import HwSimPythonAnalysis as analysis_module
        self.analysis = analysis_module
        self.args, self.inputs = args, validate_study(inputs)
        self.batch, self.output = args.batch.resolve(), args.output_root.resolve()
        self.path = self.batch/'status.json'
        self.lock = threading.Lock()
        hashes = {str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for pattern in ('*.py', 'scripts/*.py', 'tests/*.py', 'launchers/*/*.py')
                  for p in sorted(CODE.glob(pattern))}
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'format_version': 3, 'status': 'prepared', 'phases': {}, 'jobs': {},
            'comparisons': {}, 'sources': {}, 'smoke_sources': {}, 'inputs': inputs,
            'analysis_version': self.analysis.ANALYSIS_VERSION, 'code_sha256': hashes,
        }
        if self.state['inputs'] != inputs or self.state['code_sha256'] != hashes:
            raise RuntimeError('Batch inputs or frozen code changed; preserve this batch')
        provenance_path = self.batch/'source-provenance.json'
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text())
            for name, expected in provenance['code_sha256'].items():
                if hashlib.sha256((CODE/name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f'Frozen source changed: {name}')
            for name, expected in provenance['manifest_sha256'].items():
                path = self.inputs['scenarios'][name]['manifest']
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f'Frozen manifest changed: {name}')

    def update(self, label=None, **fields):
        with self.lock:
            target = self.state['jobs'].setdefault(label, {}) if label else self.state
            target.update(fields)
            write_json(self.path, self.state)
            print(label or 'batch', json.dumps(fields), flush=True)

    def invoke(self, label, arguments):
        previous = self.state['jobs'].get(label, {})
        if previous.get('phase') == 'complete':
            directory = Path(previous['run'])
            assert_completed(directory)
            return directory
        if previous:
            raise RuntimeError(f'{label} has recoverable state. Inspect its logs/checkpoints; '
                               'the launcher will not restart it or remove stored events.')
        command = [sys.executable, str(CODE/'HwSimPythonAnalysis.py'), *arguments,
                   '--output-root', str(self.output), '--run-name', label]
        write_json(self.batch/f'{label}.command.json', command)
        self.update(label, phase='running', command=command, started_utc=datetime.now(timezone.utc).isoformat())
        try:
            logfile = self.batch/f'{label}.log'
            with logfile.open('x') as log:
                process = subprocess.Popen(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT)
                self.update(label, pid=process.pid)
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            directory = Path(logfile.read_text().strip().splitlines()[-1])
            assert_completed(directory)
            self.analysis.validate_html_links(directory/'index.html')
            if not (directory/'summaries/asymmetries.json').is_file():
                raise RuntimeError(f'New asymmetry summaries are absent: {directory}')
            self.update(label, phase='complete', run=str(directory))
            return directory
        except BaseException as error:
            self.update(label, phase='failed', error=str(error))
            raise

    def event_run(self, name, profile, smoke=False):
        label = analysis_label(self.inputs, name, profile, 'uncut', smoke)
        directory = self.invoke(label, event_arguments(self.inputs, name, profile, 'uncut', smoke))
        config, results, metadata = self.analysis.load_completed_run(directory)
        assert config.cut_profile == profile and config.jet_acceptance_variable == 'pseudorapidity'
        assert config.pull_magnitude_min is None and config.particle_pt_min_gev == .1
        assert config.scenario.identifier == self.inputs['scenarios'][name]['identifier']
        assert metadata['analysis_version'] == self.analysis.ANALYSIS_VERSION
        assert len(results) == 5 and bool(metadata['partial']) == smoke
        for result in results:
            assert result.processed_entries == (min(self.inputs['smoke_events'], result.total_entries) if smoke else result.total_entries)
            assert result.invalid_events == 0
            assert set(result.pull_asymmetry_moments) == set(self.inputs['asymmetry_observables'])
            for moments in result.pull_asymmetry_moments.values():
                moments.validate()
        self.analysis.validate_results(results, smoke)
        validation = {'processes': len(results), 'events': {r.spec.name:r.processed_entries for r in results}}
        if not smoke:
            baseline = self.validate_baseline(name, profile, results)
            if baseline is not None:
                validation['baseline'] = baseline
        self.update(label, validation=validation)
        with self.lock:
            target = self.state['smoke_sources'] if smoke else self.state['sources'].setdefault(profile, {}).setdefault('uncut', {})
            target[name] = str(directory)
        self.update()
        return directory

    def validate_baseline(self, name, profile, results):
        """Current uncut observables must reproduce the previous completed study."""
        import numpy as np
        path = Path(self.inputs.get('baseline_status', ''))
        if not path.is_file():
            return None
        state = json.loads(path.read_text())
        source = state.get('sources', {}).get(profile, {}).get('uncut', {}).get(name)
        if source is None:
            return None
        _, old_results, _ = self.analysis.load_completed_run(Path(source))
        old = {r.spec.name:r for r in old_results if r.strategy == 'cutbased'}
        for result in results:
            previous = old[result.spec.name]
            assert result.processed_entries == previous.processed_entries
            np.testing.assert_allclose(result.generated_sumw, previous.generated_sumw, rtol=1.e-12)
            for step, stat in result.cutflow.items():
                before = previous.cutflow[step]
                assert stat.raw_count == before.raw_count
                np.testing.assert_allclose([stat.sumw, stat.sumw2], [before.sumw, before.sumw2], rtol=1.e-9, atol=1.e-7)
            for key, hist in result.histograms.items():
                before = previous.histograms[key]
                np.testing.assert_allclose(hist.sumw, before.sumw, rtol=1.e-9, atol=1.e-7)
                np.testing.assert_allclose(hist.sumw2, before.sumw2, rtol=1.e-9, atol=1.e-7)
        return {'run': str(source), 'cutflows_and_existing_observables': 'unchanged'}

    def comparisons(self, profile, runs, smoke=False):
        for group in self.inputs['comparison_sets']:
            label = comparison_label(self.inputs, profile, 'uncut', group, smoke)
            directory = self.invoke(label, comparison_arguments(self.inputs, group, runs))
            metadata = assert_completed(directory)
            assert [s['run_id'] for s in metadata['source_runs']] == [
                runs[name].name for name in self.inputs['comparison_sets'][group]]
            key = comparison_key(profile, 'uncut', group)
            if smoke:
                key = 'smoke/' + key
            with self.lock:
                self.state['comparisons'][key] = str(directory)
            self.update()

    def event_group(self, profile, smoke=False):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.inputs['concurrent_analyses']) as pool:
            futures = {name:pool.submit(self.event_run, name, profile, smoke) for name in self.inputs['scenarios']}
            return {name:future.result() for name,future in futures.items()}

    def execute(self):
        phase = self.args.phase
        if self.state['phases'].get(phase) == 'complete':
            print(f'{phase} already complete; existing runs preserved.', flush=True)
            return
        prerequisite = {'analysis':'smoke', 'plots':'analysis'}.get(phase)
        if prerequisite and self.state['phases'].get(prerequisite) != 'complete':
            raise RuntimeError(f'Run {prerequisite} successfully before {phase}')
        self.state['phases'][phase] = 'running'
        self.update(status=phase)
        try:
            if phase == 'smoke':
                profile = self.inputs['smoke_profile']
                runs = self.event_group(profile, smoke=True)
                self.comparisons(profile, runs, smoke=True)
            elif phase == 'analysis':
                for profile in self.inputs['profiles']:
                    self.event_group(profile)
            else:
                for profile in self.inputs['profiles']:
                    runs = {name:Path(path) for name,path in self.state['sources'][profile]['uncut'].items()}
                    self.comparisons(profile, runs)
            self.state['phases'][phase] = 'complete'
            self.update(status='plots-complete' if phase == 'plots' else phase+'-complete')
        except BaseException as error:
            self.state['phases'][phase] = 'failed'
            self.update(status='failed', error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'analysis', 'plots'), required=True)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    with (args.batch/'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another study controller is live; leave it running')
        Study(args, json.loads(args.inputs.read_text())).execute()


if __name__ == '__main__':
    main()
