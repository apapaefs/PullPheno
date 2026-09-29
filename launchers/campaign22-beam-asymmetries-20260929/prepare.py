#!/usr/bin/env python3
"""Read-only preflight or immutable preparation; never process event samples."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
TAG = 'campaign22-beam-asymmetries-20260929'
DEFAULT_REPOSITORY = Path('/home/apapaefs/Projects/PullPheno')
DEFAULT_OUTPUT = Path('/micron/Projects/PullPheno/analysis-results') / TAG
STUDY_PATH = f'studies/{TAG}.json'
BR_HGG = .002270


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_identical_or_new(path, content):
    """Idempotent preparation cannot silently replace frozen configuration."""
    path = Path(path)
    content = content if isinstance(content, str) else json.dumps(content, indent=2) + '\n'
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f'Refusing to replace existing preparation: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(content)


def git(repository, *arguments, binary=False):
    return subprocess.check_output(['git', '-C', str(repository), *arguments],
                                   text=not binary).strip() if not binary else subprocess.check_output(
                                       ['git', '-C', str(repository), *arguments])


def committed_inputs(repository, revision):
    return json.loads(git(repository, 'show', f'{revision}:{STUDY_PATH}'))


def check(repository, output, root_check=True):
    revision = git(repository, 'rev-parse', 'HEAD')
    inputs = committed_inputs(repository, revision)
    required = ['HwSimPythonAnalysis.py', 'beam_asymmetry_plots.py',
                'scripts/run_beam_asymmetry_study.py', 'scripts/build_beam_asymmetry_report.py']
    for name in required:
        git(repository, 'cat-file', '-e', f'{revision}:{name}')
    runtime = inputs['runtime']
    for name in ('root_setup', 'python', 'thermal_guard', 'thermal_config'):
        if not Path(runtime[name]).is_file():
            raise FileNotFoundError(f'Missing {name}: {runtime[name]}')
    import numpy  # noqa: F401
    import fastjet  # noqa: F401
    import matplotlib  # noqa: F401
    import ROOT
    ROOT.gROOT.SetBatch(True)
    manifests, inventory = {}, []
    for name, scenario in inputs['scenarios'].items():
        source = Path(scenario['manifest'])
        manifest = json.loads(source.read_text())
        if manifest['scenario']['id'] != scenario['identifier']:
            raise RuntimeError(f'Scenario identity differs: {source}')
        if {s['name'] for s in manifest['samples']} != {'VBFH', 'QCDHjj', 'aajj', 'VBFZ', 'QCDZjj'}:
            raise RuntimeError(f'Expected five processes in {source}')
        if name == 'mg5-pythia' and manifest['scenario']['parameters'].get('SpaceShower:dipoleRecoil') is not True:
            raise RuntimeError('Older Pythia inputs must use dipole recoil')
        if name != 'mg5-pythia':
            parameters = manifest['scenario']['parameters']
            if parameters.get('H_Z_campaign') != 22 or parameters.get('aajj_campaign') != 14:
                raise RuntimeError('Herwig inputs must use campaign 22 H/Z and campaign 14 diphoton background')
        manifests[name] = manifest
        for sample in manifest['samples']:
            if name != 'mg5-pythia' and sample.get('campaign_id') != (14 if sample['name'] == 'aajj' else 22):
                raise RuntimeError(f'Unexpected sample campaign: {name}/{sample["name"]}')
            if sample['cross_section_pb'] <= 0:
                raise RuntimeError(f'Invalid normalization: {name}/{sample["name"]}')
            for filename in sample['files']:
                path = Path(filename)
                if not path.is_file() or not str(path).startswith('/micron/'):
                    raise FileNotFoundError(f'Cached ROOT input unavailable: {path}')
                item = {'scenario': name, 'sample': sample['name'], 'path': str(path),
                        'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
                if root_check:
                    file = ROOT.TFile.Open(str(path), 'READ')
                    if not file or file.IsZombie() or file.TestBit(ROOT.TFile.kRecovered):
                        raise RuntimeError(f'Unreadable ROOT input: {path}')
                    try:
                        tree = file.Get(manifest.get('tree_name', 'Data'))
                        if not tree or not {'numparticles', 'objects', 'evweight'} <= {
                                b.GetName() for b in tree.GetListOfBranches()}:
                            raise RuntimeError(f'Input tree/branches unavailable: {path}')
                        item['entries'] = int(tree.GetEntries())
                        if item['entries'] <= 0:
                            raise RuntimeError(f'Empty input: {path}')
                        expected = sample.get('reshower_validation', {}).get('entries')
                        if len(sample['files']) == 1 and expected is not None and item['entries'] != expected:
                            raise RuntimeError(f'Cached Pythia entries differ from retained reshower validation: {path}')
                        for entry in (0, item['entries']//2, item['entries']-1):
                            if tree.GetEntry(entry) <= 0:
                                raise RuntimeError(f'Unreadable entry {entry}: {path}')
                    finally:
                        file.Close()
                inventory.append(item)
    nominal = {s['name']: s for s in manifests['herwig']['samples']}
    for name, sample in nominal.items():
        if name in ('VBFH', 'QCDHjj'):
            expected = sample['generator_cross_section_pb'] * BR_HGG
            if abs(sample['cross_section_pb']-expected) > 1.e-10*expected:
                raise RuntimeError(f'Higgs branching fraction must be applied once: {name}')
    for sample in manifests['preco025']['samples']:
        if sample['cross_section_pb'] != nominal[sample['name']]['cross_section_pb']:
            raise RuntimeError('Herwig pReco must retain nominal reference cross sections')
    batch = output / 'batches' / TAG
    if (batch/'source-provenance.json').exists():
        provenance = json.loads((batch/'source-provenance.json').read_text())
        if provenance['commit'] != revision:
            raise RuntimeError('Existing batch belongs to another commit; preserve it and choose a new output root')
        for name, value in provenance['code_sha256'].items():
            if digest(batch/'code'/name) != value:
                raise RuntimeError(f'Frozen code was modified: {name}')
    if shutil.disk_usage(output if output.exists() else output.parent).free < 20*1024**3:
        raise RuntimeError('At least 20 GiB free space is required for analysis artifacts')
    receipt = {'ready': True, 'commit': revision, 'root_version': ROOT.gROOT.GetVersion(),
               'cached_inputs': inventory, 'fresh_analyses': 9, 'comparisons': 9,
               'maximum_event_workers': 380, 'batch': str(batch),
               'report': str(output/'report/index.html'), 'event_processing_started': False}
    print(json.dumps(receipt, indent=2), flush=True)
    return inputs, manifests, receipt


def prepare(repository, output, inputs, manifests, receipt):
    batch = output/'batches'/TAG
    revision = receipt['commit']
    archive = git(repository, 'archive', '--format=tar', revision, binary=True)
    code = batch/'code'
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or '..' in relative.parts or not (member.isfile() or member.isdir()):
                raise RuntimeError(f'Unsupported source archive entry: {member.name}')
            target = code/relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            content = stream.extractfile(member).read()
            if target.exists():
                if target.read_bytes() != content:
                    raise RuntimeError(f'Frozen source differs: {target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as destination:
                    destination.write(content)
                target.chmod(member.mode)
    frozen = copy.deepcopy(inputs)
    frozen['output_root'] = str(output)
    for name, scenario in frozen['scenarios'].items():
        destination = batch/'manifests'/f'{name}.json'
        write_identical_or_new(destination, manifests[name])
        scenario['manifest'] = str(destination)
    hashes = {str(p.relative_to(code)): digest(p) for p in sorted(code.rglob('*')) if p.is_file()}
    provenance = {'commit': revision, 'repository': str(repository.resolve()),
                  'remote': git(repository, 'remote', 'get-url', 'origin'),
                  'archive_sha256': hashlib.sha256(archive).hexdigest(), 'code_sha256': hashes,
                  'manifest_sha256': {name: digest(s['manifest']) for name,s in frozen['scenarios'].items()},
                  'source_manifest_sha256': {name: digest(s['manifest']) for name,s in inputs['scenarios'].items()},
                  'input_inventory': receipt['cached_inputs'], 'event_processing_started': False}
    normalization = {'higgs_branching_ratio': BR_HGG,
                     'common_reference': 'Campaign 22 nominal Herwig H/Z; campaign 14 continuum diphoton background',
                     'reference_cross_sections_pb': {s['name']:s['cross_section_pb'] for s in manifests['herwig']['samples']},
                     'manifests': manifests}
    write_identical_or_new(batch/'inputs.json', frozen)
    write_identical_or_new(batch/'source-provenance.json', provenance)
    write_identical_or_new(batch/'sample-normalization.json', normalization)
    write_identical_or_new(batch/'preflight.json', receipt)
    thermal = ('. "${HOME}/.config/pullpheno-thermal-guard.conf"\n'
               'THERMAL_GUARD_CONTROLLER_SCRIPT=run_beam_asymmetry_study.py\n'
               'THERMAL_GUARD_CONTROLLER_ACTION=--phase\n'
               'THERMAL_GUARD_INCLUDE_DESCENDANT_GROUPS=1\n'
               f'THERMAL_GUARD_STATE_DIR={batch}/thermal-guard\n')
    write_identical_or_new(batch/'thermal-guard.conf', thermal)
    (output/'runs').mkdir(parents=True, exist_ok=True)
    print(f'Prepared {batch}; event processing has not been started.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    values = check(args.repository, args.output_root)
    if args.prepare:
        prepare(args.repository, args.output_root, *values)


if __name__ == '__main__':
    main()
