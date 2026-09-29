#!/usr/bin/env python3
"""Validate and build dedicated campaign-22 asymmetry galleries and summaries."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import numpy as np  # noqa: E402
import HwSimPythonAnalysis as analysis  # noqa: E402
from scripts.build_pullcut_report import inline_plot, page  # noqa: E402
from scripts.build_pullcut_scan_report import validate_embedded  # noqa: E402
from scripts.pullcut_study_config import comparison_key  # noqa: E402
from scripts.run_beam_asymmetry_study import validate_study, write_json  # noqa: E402

CHANNELS = (('higgs', 'H → γγ'), ('z', 'Z → e⁺e⁻, μ⁺μ⁻'))
OBSERVABLES = {'pull_asymmetry_beam': 'Beam pull asymmetry', 'pull_asymmetry_phi': 'Azimuthal pull asymmetry'}
ETAS = {'eta2-mll71-111': 2., 'eta3-mll71-111': 3., 'eta45-mll71-111': 4.5}


def validate_scaling(payload, luminosities):
    """Central A is common to luminosities; projected covariance is proportional to 1/L."""
    low = min(luminosities)
    checked = 0
    for prediction in payload['predictions']:
        for key, values in prediction['observables'].items():
            if values['A'] is None:
                continue
            total = np.asarray(values['A'])
            contributions = values.get('process_contributions', {})
            if contributions:
                np.testing.assert_allclose(sum(np.asarray(v) for v in contributions.values()), total,
                                           rtol=2.e-12, atol=1.e-16)
            covariance = np.asarray(values['mc_covariance'])
            assert covariance.shape == (len(total), len(total))
            assert np.all(np.isfinite(covariance))
            np.testing.assert_allclose(covariance, covariance.T, atol=1.e-14)
            first = values['data_covariances'][str(float(low))]
            for luminosity in luminosities:
                np.testing.assert_allclose(values['data_covariances'][str(float(luminosity))],
                                           np.asarray(first)*low/luminosity, rtol=2.e-10, atol=1.e-18)
            checked += 1
    for ratio in payload.get('ratios', []):
        valid = np.asarray(ratio['valid'], dtype=bool)
        central = np.asarray(ratio['central'], dtype=float)
        numerical = ratio.get('numerical_valid')
        assert valid.shape == central.shape
        denominator_valid = np.asarray(ratio.get('denominator_valid', valid), dtype=bool)
        assert denominator_valid.shape == valid.shape
        assert np.all(~valid | denominator_valid)
        if numerical is None:
            assert np.array_equal(valid, np.isfinite(central))
        else:
            # Nonzero denominators stay physically valid even if a ratio or
            # its covariance exceeds float64. Each error model has its own
            # numerical support, since the MC reference term can overflow
            # while the projected-data covariance remains representable.
            assert np.all(~np.isfinite(central[~valid]))
        models = [('mc', ratio.get('mc_covariance'),
                   valid if numerical is None else numerical['mc'])]
        models += [(lumi, covariance, valid if numerical is None else numerical['data'][lumi])
                   for lumi,covariance in ratio.get('data_covariances', {}).items()]
        support = {}
        for model, covariance, mask in models:
            if covariance is None:
                continue
            covariance = np.asarray(covariance, dtype=float)
            accepted = np.asarray(mask, dtype=bool)
            assert accepted.shape == valid.shape
            assert covariance.shape == (len(valid), len(valid))
            assert np.all(~accepted | (valid & np.isfinite(central)))
            if numerical is not None:
                np.testing.assert_array_equal(accepted, valid & np.isfinite(central) &
                                              np.isfinite(np.diag(covariance)))
            assert np.all(np.isfinite(covariance[np.ix_(accepted, accepted)]))
            assert np.all(~np.isfinite(covariance[~valid, :]))
            assert np.all(~np.isfinite(covariance[:, ~valid]))
            np.testing.assert_allclose(covariance, covariance.T, equal_nan=True)
            support[model] = accepted, covariance
        first = support.get(str(float(low)))
        if first is not None:
            for luminosity in luminosities:
                high = support.get(str(float(luminosity)))
                if high is not None:
                    shared = first[0] & high[0]
                    indices = np.ix_(shared, shared)
                    np.testing.assert_allclose(high[1][indices], first[1][indices]*low/luminosity,
                                               rtol=2.e-10, atol=1.e-18)
    return checked


def validate_pairwise(full, subset):
    predictions = {(p['strategy'], p['channel'], p['scenario']):p['observables'] for p in full['predictions']}
    for prediction in subset['predictions']:
        key = prediction['strategy'], prediction['channel'], prediction['scenario']
        if prediction['observables'] != predictions[key]:
            raise AssertionError(f'Pairwise asymmetry prediction changed: {key}')
    ratios = {(r['strategy'], r['channel'], r['observable'], r['scenario']):r for r in full['ratios']}
    for ratio in subset['ratios']:
        key = ratio['strategy'], ratio['channel'], ratio['observable'], ratio['scenario']
        if ratio != ratios[key]:
            raise AssertionError(f'Pairwise asymmetry ratio changed: {key}')


def record_selection(records, **fields):
    return [r for r in records if all(r.get(name) == value for name,value in fields.items())]


def render_records(records, destination, title, note='', home='../index.html'):
    """Self-contained PNG/PDF viewing follows the established campaign report UI."""
    figures = []
    for index, (run, record) in enumerate(records):
        label = (f'{record.get("title", OBSERVABLES.get(record["observable"],record["observable"]))}; '
                 f'{record["channel"]}; {record["view"]}; {record["error_mode"]} uncertainty')
        if record.get('sample'):
            label += '; ' + record['sample']
        if record.get('profile') in ETAS:
            label += f'; |ηj| < {ETAS[record["profile"]]:g}'
        name = f'{destination.stem}-{index}-{Path(record["png"]).stem}'
        figures.append('<figure>'+inline_plot(run/record['png'], run/record['pdf'], label, name)+
                       '<figcaption>'+html.escape(label)+'</figcaption></figure>')
    document = page(title, '<nav><a href="'+home+'">Study overview</a></nav><h1>'+html.escape(title)+
                    '</h1><p class="note">'+html.escape(note)+'</p><section class="figures">'+
                    ''.join(figures)+'</section>')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document)
    return validate_embedded(destination, records)


def build(status, destination):
    state = json.loads(status.read_text())
    if state.get('phases', {}).get('plots') != 'complete':
        raise RuntimeError('Complete smoke, analysis and plots before building the report')
    inputs = validate_study(state['inputs'])
    batch = status.parent
    provenance = json.loads((batch/'source-provenance.json').read_text())
    for name, expected in provenance['code_sha256'].items():
        if hashlib.sha256((CODE/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen report source changed: {name}')
    destination.mkdir(parents=True, exist_ok=True)
    comparisons, validation, galleries = {}, {'status':'passed', 'pages':[], 'comparisons':[]}, []
    for profile in inputs['profiles']:
        for group in inputs['comparison_sets']:
            key = comparison_key(profile, 'uncut', group)
            run = Path(state['comparisons'][key])
            metadata = json.loads((run/'run.json').read_text())
            assert metadata['status'] == 'complete' and not metadata['partial']
            assert metadata['configuration']['pull_magnitude_min'] is None
            assert metadata['configuration']['particle_pt_min_gev'] == .1
            assert metadata['configuration']['cut_profile'] == profile
            assert [source['run_id'] for source in metadata['source_runs']] == [
                Path(state['sources'][profile]['uncut'][name]).name for name in inputs['comparison_sets'][group]]
            payload = json.loads((run/'summaries/asymmetries.json').read_text())
            count = validate_scaling(payload, inputs['luminosities_fb'])
            records = payload['plots']
            assert records and all(r['observable'] in OBSERVABLES for r in records)
            assert all(r['kind'].startswith('asymmetry-') for r in records)
            comparisons[(profile,group)] = run, payload, records
            target = destination/'galleries'/f'{profile}-{group}.html'
            note = f'|ηj| < {ETAS[profile]:g}; no pull-magnitude cut. Both luminosities and uncertainty models, with full and near-zero views.'
            validation['pages'].append(render_records([(run,r) for r in records], target,
                                                      inputs['comparison_labels'][group], note))
            galleries.append((target.relative_to(destination).as_posix(), profile+' · '+group))
            validation['comparisons'].append({'key':key, 'run':run.name, 'predictions_checked':count})
        full = comparisons[(profile,'three-way')][1]
        for group in inputs['comparison_sets']:
            validate_pairwise(full, comparisons[(profile,group)][1])
    individual = []
    for profile in inputs['profiles']:
        for name in inputs['scenarios']:
            run = Path(state['sources'][profile]['uncut'][name])
            config, results, metadata = analysis.load_completed_run(run)
            assert not metadata['partial'] and config.pull_magnitude_min is None
            analysis.validate_results(results, False)
            payload = json.loads((run/'summaries/asymmetries.json').read_text())
            records = payload['plots']
            assert len(payload['predictions']) == 5 and records
            assert {r['sample'] for r in records} == {'VBFH','QCDHjj','aajj','VBFZ','QCDZjj'}
            target = destination/'galleries'/f'{profile}-{name}-samples.html'
            validation['pages'].append(render_records([(run,r) for r in records], target,
                                                      f'{name}: individual process asymmetries',
                                                      f'|ηj| < {ETAS[profile]:g}; finite-MC and projected-data uncertainty at both luminosities; no pull-magnitude cut.'))
            individual.append((target.relative_to(destination).as_posix(), profile+' · '+name))
    summaries = []
    for observable, label in OBSERVABLES.items():
        for luminosity in inputs['luminosities_fb']:
            for group in inputs['comparison_sets']:
                selected = []
                for profile in inputs['profiles']:
                    run, _, records = comparisons[(profile,group)]
                    for channel, _ in CHANNELS:
                        for mode in ('mc','data'):
                            for view in ('full','near-zero'):
                                matches = record_selection(records, observable=observable, channel=channel,
                                                           luminosity_fb=luminosity, error_mode=mode,
                                                           view=view, strategy='cutbased')
                                assert len(matches) == 1, (profile,group,channel,mode,view,len(matches))
                                selected.append((run,{**matches[0], 'profile':profile}))
                target = destination/'summaries'/f'{observable}-{luminosity:g}-{group}.html'
                title = f'{label} · {luminosity:g} fb⁻¹ · {inputs["comparison_labels"][group]}'
                note = 'H and Z channels at |ηj| < 2, 3 and 4.5. Full and near-zero views for finite-MC and projected-data uncertainty. Ratios use nominal Herwig.'
                validation['pages'].append(render_records(selected, target, title, note))
                summaries.append((target.relative_to(destination).as_posix(),title))
    assert len(summaries) == 12
    def links(rows):
        return '<ul>'+''.join('<li><a href="'+html.escape(path,quote=True)+'">'+html.escape(label)+'</a></li>' for path,label in rows)+'</ul>'
    downloads = []
    for profile in inputs['profiles']:
        for group in inputs['comparison_sets']:
            run = comparisons[(profile,group)][0]
            for suffix in ('json','npz'):
                target = run/f'summaries/asymmetries.{suffix}'
                assert target.is_file()
                downloads.append((os.path.relpath(target,destination), f'{profile} · {group} · {suffix.upper()}'))
        for name in inputs['scenarios']:
            run = Path(state['sources'][profile]['uncut'][name])
            for suffix in ('json','npz'):
                target = run/f'summaries/asymmetries.{suffix}'
                assert target.is_file()
                downloads.append((os.path.relpath(target,destination), f'{profile} · {name} individual processes · {suffix.upper()}'))
    title = inputs['report_title']
    content = ('<h1>'+html.escape(title)+'</h1><p>'+html.escape(inputs['report_description'])+'</p>'
               '<p class="note">60 linear bins in |x| on [0,0.03); x = t_beam or t_phi. '
               'No pull-magnitude cut. Zero projections and overflow remain in the selected denominator. '
               'Both tagging jets have half weight. Nominal Herwig process contributions use one channel denominator and stack separately by sign.</p>'
               '<p>'+html.escape(inputs.get('report_note',''))+'</p>'
               '<p>Source commit: <code>'+html.escape(provenance['commit'])+'</code>. '
               'Nine fresh ROOT analyses and nine comparisons; older Pythia event files were reanalysed.</p>'
               '<section><h2>Asymmetry summaries</h2>'+links(summaries)+'</section>'
               '<section><h2>Comparison galleries</h2>'+links(galleries)+'</section>'
               '<section><h2>Individual process galleries</h2>'+links(individual)+'</section>'
               '<section><h2>Numerical exports</h2>'+links(downloads)+'</section>'
               '<footer><a href="sample-normalization.json">Sample normalization</a> · '
               '<a href="source-provenance.json">Source and input provenance</a> · '
               '<a href="validation.json">Validation</a></footer>')
    (destination/'index.html').write_text(page(title,content))
    for name in ('sample-normalization.json','source-provenance.json','inputs.json'):
        (destination/name).write_text((batch/name).read_text())
    validation.update(summary_pages=len(summaries), comparison_galleries=len(galleries), sample_galleries=len(individual))
    write_json(destination/'validation.json',validation)
    for path in destination.rglob('*.html'):
        analysis.validate_html_links(path)
    state['status'] = 'complete'
    state['phases']['report'] = 'complete'
    state['report'] = str(destination/'index.html')
    write_json(status,state)
    print(destination/'index.html',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args = parser.parse_args()
    build(args.status,args.destination)


if __name__ == '__main__':
    main()
