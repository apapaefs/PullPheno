#!/usr/bin/env python3
"""Run the fixed three-scenario pull-cut study in an isolated code snapshot.

Requires an explicit smoke phase before production. Existing completed runs and
nonempty checkpoint directories are never modified or removed by this driver.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import numpy as np
import HwSimPythonAnalysis as analysis

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--phase', choices=('smoke', 'production'), required=True)
parser.add_argument('--batch', type=Path, required=True)
parser.add_argument('--inputs', type=Path, required=True)
parser.add_argument('--output-root', type=Path, required=True)
args = parser.parse_args()
BATCH = args.batch.resolve()
OUTPUT = args.output_root.resolve()
INPUTS = json.loads(args.inputs.read_text())
if 'selections' in INPUTS:
    from scripts.run_pullcut_scan import run
    run(args, INPUTS)
    sys.exit(0)
NAMES = tuple(INPUTS['scenarios'])
PARTICLE_PT_MIN = INPUTS.get('particle_pt_min_gev')
STUDY_TAG = INPUTS.get('study_tag', 'v' + analysis.ANALYSIS_VERSION.replace('.', '')[:2])
STATE_PATH = BATCH / 'status.json'
LOCK = threading.Lock()
STATE = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {
    'status': 'new', 'jobs': {}, 'comparisons': {}, 'inputs': INPUTS,
    'analysis_version': analysis.ANALYSIS_VERSION,
    'study_tag': STUDY_TAG, 'particle_pt_min_gev': PARTICLE_PT_MIN,
    'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in CODE.glob('*.py')},
}


def update(key=None, **fields):
    with LOCK:
        target = STATE['jobs'].setdefault(key, {}) if key else STATE
        target.update(fields)
        temp = STATE_PATH.with_suffix('.tmp')
        temp.write_text(json.dumps(STATE, indent=2) + '\n')
        temp.replace(STATE_PATH)
        print(key or 'batch', json.dumps(fields), flush=True)


def invoke(label, arguments):
    previous = STATE['jobs'].get(label, {})
    if previous.get('phase') == 'complete':
        run = Path(previous['run'])
        assert json.loads((run/'run.json').read_text())['status'] == 'complete'
        return run
    if previous.get('phase') in ('running', 'failed'):
        raise RuntimeError(f'{label} has recoverable state; inspect its log/checkpoints before resuming')
    command = [sys.executable, str(CODE/'HwSimPythonAnalysis.py'), *arguments,
               '--output-root', str(OUTPUT), '--run-name', label]
    (BATCH/f'{label}.command.json').write_text(json.dumps(command, indent=2)+'\n')
    update(label, phase='running', command=command)
    try:
        with (BATCH/f'{label}.log').open('x') as log:
            subprocess.run(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT, check=True)
        run = Path((BATCH/f'{label}.log').read_text().strip().splitlines()[-1])
        metadata = json.loads((run/'run.json').read_text())
        assert metadata['status'] == 'complete'
        analysis.validate_html_links(run/'index.html')
        if metadata['run_type'] == 'analysis':
            config, results, _ = analysis.load_completed_run(run)
            analysis.validate_results(results, metadata['partial'])
            validation = {r.spec.name: {
                'processed_entries': r.processed_entries, 'generated_sumw': r.generated_sumw,
                'invalid_events': r.invalid_events, **analysis.pull_selection_summary(r),
            } for r in results}
        else:
            validation = {'plot_count': len(json.loads((run/'summaries/plots.json').read_text()))}
        update(label, phase='complete', run=str(run), validation=validation)
        return run
    except Exception as error:
        update(label, phase='failed', error=str(error))
        raise


def event_run(name, profile, selection, smoke=False):
    label = f'{name}-{profile}-{selection}-{STUDY_TAG}' + ('-smoke' if smoke else '')
    arguments = ['--analyses', 'cutbased', '--luminosities', '300', '3000']
    if PARTICLE_PT_MIN is None and selection == 'uncut' and profile in INPUTS.get('uncut_sources', {}) and not smoke:
        source = OUTPUT/'runs'/INPUTS['uncut_sources'][profile][name]
        arguments += ['--from-run', str(source)]
    else:
        arguments += ['--samples', INPUTS['scenarios'][name]['manifest'], '--cut-profile', profile]
        if PARTICLE_PT_MIN is not None:
            arguments += ['--particle-pt-min', str(PARTICLE_PT_MIN)]
        if selection == 'pullcut-0p0015':
            arguments += ['--pull-magnitude-min', '0.0015']
        arguments += ['--workers', '4' if smoke else '112', '--event-shards', '4' if smoke else '256']
        if smoke:
            arguments += ['--max-events', '5000']
    run = invoke(label, arguments)
    config, results, metadata = analysis.load_completed_run(run)
    assert config.cut_profile == profile and config.jet_acceptance_variable == 'pseudorapidity'
    assert config.pull_magnitude_min == (.0015 if selection != 'uncut' else None)
    assert config.particle_pt_min_gev == PARTICLE_PT_MIN
    assert len(results) == 5
    assert bool(metadata['partial']) == smoke
    assert all(r.processed_entries == (min(5000, r.total_entries) if smoke else r.total_entries) for r in results)
    if not smoke:
        baseline_id = (INPUTS.get('uncut_sources', {}).get(profile, {}).get(name)
                       if PARTICLE_PT_MIN is None else None)
        if baseline_id is None and selection != 'uncut':
            baseline_id = Path(STATE['jobs'][f'{name}-{profile}-uncut-{STUDY_TAG}']['run']).name
        if baseline_id:
            old_config, old_results, _ = analysis.load_completed_run(OUTPUT/'runs'/baseline_id)
            old_by_name = {r.spec.name:r for r in old_results if r.strategy == 'cutbased'}
            for result in results:
                old = old_by_name[result.spec.name]
                assert result.processed_entries == old.processed_entries
                np.testing.assert_allclose(result.generated_sumw, old.generated_sumw, rtol=1e-12)
                for step, stat in result.cutflow.items():
                    assert stat.raw_count == old.cutflow[step].raw_count
                    np.testing.assert_allclose([stat.sumw, stat.sumw2],
                                               [old.cutflow[step].sumw, old.cutflow[step].sumw2], rtol=1e-9, atol=1e-7)
                for key, hist in result.histograms.items():
                    if selection == 'uncut' or key not in analysis.PULL_OBSERVABLE_KEYS:
                        np.testing.assert_allclose(hist.sumw, old.histograms[key].sumw, rtol=1e-9, atol=1e-7)
                assert result.pull_total_sumw <= old.pull_total_sumw * (1+1e-10)
    baseline_id = INPUTS.get('baseline_sources', {}).get(profile, {}).get(selection, {}).get(name)
    if not smoke and baseline_id:
        _, old_results, _ = analysis.load_completed_run(OUTPUT/'runs'/baseline_id)
        old_by_name = {r.spec.name:r for r in old_results if r.strategy == 'cutbased'}
        changes = {}
        for result in results:
            old = old_by_name[result.spec.name]
            assert result.processed_entries == old.processed_entries
            np.testing.assert_allclose(result.generated_sumw, old.generated_sumw, rtol=1e-12)
            equal = all(np.allclose(h.sumw, old.histograms[k].sumw, rtol=1e-9, atol=1e-7)
                        for k,h in result.histograms.items())
            equal_counts = all(c.raw_count == old.cutflow[k].raw_count for k,c in result.cutflow.items())
            changes[result.spec.name] = {
                'histograms_unchanged': equal, 'cutflow_counts_unchanged': equal_counts,
                'old_selected_sumw': old.cutflow['boson_centrality'].sumw,
                'new_selected_sumw': result.cutflow['boson_centrality'].sumw,
                'old_accepted_pull_sumw': old.pull_total_sumw,
                'new_accepted_pull_sumw': result.pull_total_sumw,
            }
            if name == 'mg5-pythia':
                assert equal and equal_counts, ('Pythia baseline changed', result.spec.name, changes[result.spec.name])
        update(label, baseline_run=baseline_id, baseline_comparison=changes)
    return run


def group(profile, selection, smoke=False):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {name:pool.submit(event_run, name, profile, selection, smoke) for name in NAMES}
        runs = [futures[name].result() for name in NAMES]
    label = f'c14-c20-{profile}-{selection}-comparison-{STUDY_TAG}' + ('-smoke' if smoke else '')
    comparison = invoke(label, ['--compare-runs', *map(str, runs), '--analyses', 'cutbased',
                               '--luminosities', '300', '3000', '--scenario-color', 'mg5-pythia8=#8A5FBF'])
    with LOCK:
        STATE['comparisons'][label] = str(comparison)
    update()


def main():
    if args.phase == 'smoke':
        assert STATE['status'] in ('new', 'smoke')
        update(status='smoke')
        group('eta45-mll71-111', 'uncut', smoke=True)
        group('eta45-mll71-111', 'pullcut-0p0015', smoke=True)
        update(status='smoke-complete')
    else:
        assert STATE['status'] in ('smoke-complete', 'production')
        update(status='production')
        for profile in ('eta45-mll71-111', 'eta2-mll71-111', 'eta3-mll71-111'):
            group(profile, 'uncut')
            group(profile, 'pullcut-0p0015')
        update(status='complete')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        update(status='failed', error=str(error))
        raise
