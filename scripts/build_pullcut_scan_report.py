"""Validate and publish a configured pull-threshold scan from completed runs."""
import base64
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re

import numpy as np
import HwSimPythonAnalysis as analysis
from scripts.build_pullcut_report import (
    CHANNELS, ETAS, PROFILES, gallery_path, inline_plot, page, write_embedded_gallery,
)
from scripts.pullcut_study_config import comparison_key, selection_text, validate_inputs
from scripts.run_pullcut_scan import validate_threshold_pair, write_json

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_scaling(observables, luminosities):
    low = min(luminosities)
    for observable in observables.values():
        reference = observable['luminosities'][str(float(low))]
        for luminosity in luminosities:
            scale = luminosity / low
            for scenario, first in reference.items():
                high = observable['luminosities'][str(float(luminosity))][scenario]
                for field, factor in (('total_yield', scale), ('data_statistical_covariance', scale),
                                      ('mc_statistical_covariance', scale**2)):
                    np.testing.assert_allclose(high[field], factor*np.asarray(first[field]), rtol=2e-12, atol=1e-10)
                shape, covariance = analysis.normalized_binned_prediction(
                    np.asarray(first['total_yield']), np.asarray(first['data_statistical_covariance']))
                high_shape, high_covariance = analysis.normalized_binned_prediction(
                    np.asarray(high['total_yield']), np.asarray(high['data_statistical_covariance']))
                np.testing.assert_allclose(high_shape, shape, rtol=2e-12, atol=1e-16)
                np.testing.assert_allclose(high_covariance, covariance/scale, rtol=2e-10, atol=1e-18)


def validate_pairwise(full, subset):
    """Every exported prediction must be independent of the other displayed samples."""
    for channel, _ in CHANNELS:
        first = full['analyses']['cutbased'][channel]
        second = subset['analyses']['cutbased'][channel]
        for scenario, values in second['scenarios'].items():
            assert values == first['scenarios'][scenario], (channel, scenario)
        for observable, values in second['observables'].items():
            reference = first['observables'][observable]
            assert values['bin_edges'] == reference['bin_edges']
            for luminosity, scenarios in values['luminosities'].items():
                for scenario, prediction in scenarios.items():
                    assert prediction == reference['luminosities'][luminosity][scenario], (channel, observable, scenario)
            assert values['comparisons_to_reference'] == {
                k: v for k, v in reference['comparisons_to_reference'].items()
                if k in values['comparisons_to_reference']
            }


class EmbeddedPlots(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.downloads = []
        self.viewers = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'details' and attrs.get('class') == 'plot-zoom':
            self.viewers += 1
        if tag == 'img' and attrs.get('src', '').startswith('data:'):
            self.images.append(hashlib.sha256(base64.b64decode(attrs['src'].split(',', 1)[1])).hexdigest())
        if tag == 'a' and 'download' in attrs:
            self.downloads.append((attrs['download'], hashlib.sha256(base64.b64decode(attrs['href'].split(',', 1)[1])).hexdigest()))


def validate_embedded(path, records):
    parser = EmbeddedPlots()
    parser.feed(path.read_text())
    assert parser.viewers == len(parser.images) == len(records)
    assert len(parser.downloads) == 2*len(records)
    assert len({name for name, _ in parser.downloads}) == len(parser.downloads)
    for index, (run, record) in enumerate(records):
        assert parser.images[index] == digest(run/record['png'])
        assert parser.downloads[2*index][1] == parser.images[index]
        assert parser.downloads[2*index+1][1] == digest(run/record['pdf'])
    return {'page': path.name, 'embedded_plots': len(records), 'downloads': 2*len(records)}


def build(args, state):
    inputs = validate_inputs(state['inputs'])
    report_title = inputs.get('report_title', 'Pull-threshold scan and generator comparisons')
    report_description = inputs.get('report_description', 'Nominal Herwig, Herwig pReco = 0.25, and MG5+Pythia 8')
    report_note = inputs.get('report_note', '')
    fresh_sources = sum(name not in inputs.get('reuse_sources', {}).get(profile, {}).get(selection, {})
                        for profile in inputs['profiles'] for selection in inputs['selections']
                        for name in inputs['scenarios'])
    fresh_comparisons = sum(group not in inputs.get('reuse_comparisons', {}).get(profile, {}).get(selection, {})
                            for profile in inputs['profiles'] for selection in inputs['selections']
                            for group in inputs['comparison_sets'])
    total_sources = len(inputs['profiles'])*len(inputs['selections'])*len(inputs['scenarios'])
    total_comparisons = len(inputs['profiles'])*len(inputs['selections'])*len(inputs['comparison_sets'])
    work_summary = (f'{fresh_sources} new ROOT analyses and {fresh_comparisons} new comparison runs; '
                    f'{total_sources-fresh_sources} archived source runs and '
                    f'{total_comparisons-fresh_comparisons} archived comparisons reused.')
    assert state['status'] == 'complete'
    results, destination = args.results.resolve(), args.destination.resolve()
    operation = args.status.parent
    preserved = json.loads((args.preserved_hashes or operation/'preserved-files.json').read_text())
    for name, checksum in preserved.items():
        assert digest(results/name) == checksum, f'Original output changed: {name}'
    for name in ('HwSimPythonAnalysis.py', 'xgboost_root_varfiles_module.py'):
        assert digest(ROOT/name) == state['code_sha256'][name]
    tests = json.loads((operation/'unit-tests.json').read_text())
    assert tests['status'] == 'passed' and tests['analysis_sha256'] == digest(ROOT/'HwSimPythonAnalysis.py')
    resume = json.loads((operation/'resume-validation.json').read_text())
    assert resume['status'] == 'passed' and not resume['root_input_reread']
    destination.mkdir(exist_ok=True)
    comparisons, sources, retention = {}, {}, []
    luminosities = inputs['luminosities_fb']
    scenario_names = {value['identifier']: name for name, value in inputs['scenarios'].items()}
    eta_values = dict(zip(PROFILES, ETAS))
    ordered_selections = sorted(inputs['selections'], key=lambda s: inputs['selections'][s] or 0.)
    full_group = next(group for group, members in inputs['comparison_sets'].items()
                      if set(members) == set(inputs['scenarios']))
    validation = {'status': 'passed', 'preserved_files': len(preserved), 'comparisons': [], 'pages': []}
    for profile in inputs['profiles']:
        for selection in ordered_selections:
            threshold = inputs['selections'][selection]
            for group, members in inputs['comparison_sets'].items():
                key = comparison_key(profile, selection, group)
                run = results/'runs'/Path(state['comparisons'][key]).name
                metadata = json.loads((run/'run.json').read_text())
                config = metadata['configuration']
                assert metadata['status'] == 'complete' and not metadata['partial']
                assert config['cut_profile'] == profile and config['jet_acceptance_variable'] == 'pseudorapidity'
                assert config['pull_magnitude_min'] == threshold
                assert config['particle_pt_min_gev'] == inputs['particle_pt_min_gev']
                assert config['luminosities_fb'] == luminosities
                assert [source['scenario']['identifier'] for source in metadata['source_runs']] == [inputs['scenarios'][name]['identifier'] for name in members]
                records = json.loads((run/'summaries/plots.json').read_text())
                data = json.loads((run/'summaries/comparison.json').read_text())
                comparisons[(profile, selection, group)] = (run, metadata, records, data)
                for channel, _ in CHANNELS:
                    validate_scaling(data['analyses']['cutbased'][channel]['observables'], luminosities)
                analysis.validate_html_links(run/'index.html')
                for source in metadata['source_runs']:
                    source_id = source['run_id']
                    source_name = scenario_names[source['scenario']['identifier']]
                    assert source_id == Path(state['sources'][profile][selection][source_name]).name
                    if source_id in sources:
                        continue
                    source_config, samples, source_metadata = analysis.load_completed_run(results/'runs'/source_id)
                    assert source_metadata['status'] == 'complete' and not source_metadata['partial']
                    assert source_config.pull_magnitude_min == threshold
                    assert source_config.particle_pt_min_gev == inputs['particle_pt_min_gev']
                    assert source_config.cut_profile == profile and source_config.jet_acceptance_variable == 'pseudorapidity'
                    analysis.validate_results(samples, False)
                    sources[source_id] = (source_config, samples, source_metadata)
                    for sample in samples:
                        assert sample.invalid_events == 0 and sample.processed_entries == sample.total_entries
                        if threshold is not None:
                            histogram = sample.histograms['pull_magnitude']
                            assert not np.any(histogram.sumw[histogram.edges[1:] <= threshold])
                            assert sample.zero_pull_jets == 0
                        for observable in analysis.PULL_OBSERVABLE_KEYS:
                            np.testing.assert_allclose(sample.histograms[observable].integral, sample.pull_total_sumw, rtol=1e-10)
                            np.testing.assert_allclose(sample.histograms[observable].sumw2.sum(),
                                sum(k*stat.sumw2/4 for k, stat in sample.pull_pass_counts.items()), rtol=1e-10)
                        stat = analysis.pull_selection_summary(sample)
                        factor = config['reference_cross_sections_pb'][sample.spec.name]/sample.generated_sumw
                        selected = sample.cutflow['boson_centrality']
                        retention.append({'profile': profile, 'selection': selection, 'pull_magnitude_min': threshold,
                            'particle_pt_min_gev': inputs['particle_pt_min_gev'], 'scenario': source['scenario']['identifier'],
                            'sample': sample.spec.name, 'channel': sample.spec.channel, 'source_run': source_id,
                            'reference_cross_section_pb': config['reference_cross_sections_pb'][sample.spec.name],
                            'selected_cross_section_pb': factor*selected.sumw,
                            'accepted_pull_cross_section_pb': factor*sample.pull_total_sumw, **stat,
                            'expected_selected_events': {str(l): 1000*l*factor*selected.sumw for l in luminosities},
                            'expected_accepted_pull_entries_half_weight': {str(l): 1000*l*factor*sample.pull_total_sumw for l in luminosities},
                            'expected_events_by_accepted_jet_count': {str(l): {k: 1000*l*factor*v['sumw'] for k,v in stat['pass_counts'].items()} for l in luminosities}})
                validation['comparisons'].append({'key': key, 'run': run.name, 'status': 'passed'})
            full = comparisons[(profile, selection, full_group)]
            for group in inputs['comparison_sets']:
                subset = comparisons[(profile, selection, group)]
                assert subset[1]['configuration']['reference_cross_sections_pb'] == full[1]['configuration']['reference_cross_sections_pb']
                validate_pairwise(full[3], subset[3])
        for name in inputs['scenarios']:
            for before, after in zip(ordered_selections, ordered_selections[1:]):
                first = Path(state['sources'][profile][before][name]).name
                second = Path(state['sources'][profile][after][name]).name
                validate_threshold_pair(sources[first][1], sources[second][1])

    expected_comparisons = len(inputs['profiles'])*len(ordered_selections)*len(inputs['comparison_sets'])
    assert len(comparisons) == expected_comparisons
    assert len(sources) == len(inputs['profiles'])*len(ordered_selections)*len(inputs['scenarios'])
    for (profile, selection, group), (run, metadata, records, data) in comparisons.items():
        write_embedded_gallery(run, destination, profile, selection, records, group)
        validation['pages'].append(validate_embedded(destination/gallery_path(profile, selection, group), [(run, record) for record in records]))

    summaries = []
    convention = ('Nominal Herwig processes are stacked. Each accepted tagging jet contributes half the event weight. '
        'Expected-data errors include correlations between entries from the same event. Ratio errors hold the nominal prediction fixed; '
        'its band shows its own counting uncertainty. Finite-MC errors and shape/rate tests are available in the full galleries. '
        'Click a plot to enlarge it and download its embedded PNG or PDF; click again or press Escape to close.')
    particle_note = f'Common input-particle pT > {inputs["particle_pt_min_gev"]:g} GeV before boson selection, isolation, clustering and pull.'
    for selection in ordered_selections:
        threshold = inputs['selections'][selection]
        for group, members in inputs['comparison_sets'].items():
            group_label = inputs['comparison_labels'][group]
            style_labels = inputs.get('scenario_style_labels', {'herwig': '', 'preco025': 'Herwig pReco = 0.25: orange squares.', 'mg5-pythia': 'MG5+Pythia 8: purple triangles.'})
            style_text = html.escape(' '.join(style_labels[name] for name in members).strip())
            for luminosity in luminosities:
                name = f'summary-{luminosity:g}fb-folded-angle-{selection}-{group}'
                sections, exported, embeds = [], [], []
                for profile in inputs['profiles']:
                    eta = eta_values[profile]
                    run, metadata, records, data = comparisons[(profile, selection, group)]
                    figures = []
                    for channel, channel_label in CHANNELS:
                        matches = [r for r in records if r['kind'] == 'reference-stack-plus-total-ratio-data-stat'
                                   and r['channel'] == channel and r['luminosity_fb'] == luminosity]
                        assert len(matches) == 1
                        record = matches[0]
                        assert record['observable'] == 'folded_pull_angle' and record['ratio_reference_uncertainty'] == 'fixed'
                        viewer = inline_plot(run/record['png'], run/record['pdf'],
                            f'{channel_label}; |ηj| < {eta:g}; {selection_text(threshold)}; {group_label}; {luminosity:g} fb−1',
                            f'particlept-0p1-{profile}-{channel}-{luminosity:g}fb-{selection}-{group}-folded-angle')
                        figures.append(f'<figure><h3>{channel_label}</h3>{viewer}<figcaption>{luminosity:g} fb<sup>−1</sup> · Expected data statistics</figcaption></figure>')
                        exported.append({'profile': profile, 'channel': channel, 'comparison_run': run.name, 'plot': record,
                            'predictions': data['analyses']['cutbased'][channel]['observables']['folded_pull_angle']['luminosities'][str(float(luminosity))]})
                        embeds.append((run, record))
                    sections.append(f'<section id="{profile}"><h2>|η<sub>j</sub>| &lt; {eta:g}</h2><p>{html.escape(selection_text(threshold))}; 71 &lt; m<sub>ll</sub> &lt; 111 GeV. '
                        f'<a href="{gallery_path(profile,selection,group)}">Full gallery and statistical tables</a>.</p><div class="figures">'+''.join(figures)+'</div></section>')
                title = f'{luminosity:g} fb⁻¹ · {selection_text(threshold)} · {group_label}'
                body = f'<h1>{html.escape(title)}</h1><p>Folded pull angle · Cut-based particle-level analysis · H → γγ and Z → e⁺e⁻, μ⁺μ⁻ · 13.6 TeV.</p>'
                body += '<nav><a href="index.html">Study overview</a>'+''.join(f'<a href="#{p}">|ηj| &lt; {eta_values[p]:g}</a>' for p in inputs['profiles'])+f'<a href="{name}.json">Binned predictions</a></nav>'
                body += f'<p class="note">{convention} {style_text}</p><p class="note">{particle_note}</p>'+''.join(sections)
                if report_note and 'mg5-pythia' in members:
                    body += f'<p class="note">{html.escape(report_note)}</p>'
                body += '<footer>Common nominal-Herwig cross sections with scenario-specific efficiencies. BR(H → γγ) = 0.002270 is applied once; Z includes electron and muon decays. Generation-level m(jj) > 200 GeV and reconstructed m(jj) > 400 GeV are retained.</footer>'
                (destination/(name+'.html')).write_text(page(title, body))
                write_json(destination/(name+'.json'), {'luminosity_fb': luminosity, 'selection': selection,
                    'pull_magnitude_min': threshold, 'particle_pt_min_gev': inputs['particle_pt_min_gev'],
                    'comparison_set': group, 'plots': exported})
                validation['pages'].append(validate_embedded(destination/(name+'.html'), embeds))
                summaries.append({'name': name, 'selection': selection, 'luminosity_fb': luminosity, 'comparison_set': group})

    body = f'<h1>{html.escape(report_title)}</h1><p>{html.escape(report_description)} · H → γγ and Z → e⁺e⁻, μ⁺μ⁻ · 13.6 TeV.</p>'
    if report_note:
        body += f'<p class="note">{html.escape(report_note)}</p>'
    body += f'<p>{html.escape(work_summary)}</p>'
    body += f'<p class="note">{particle_note} All jet acceptances use pseudorapidity; the rapidity definitions of the gap, hemispheres, centrality and pull remain unchanged.</p>'
    body += '<nav><a href="../index.html">All results</a><a href="retention.json">Retention and yields</a><a href="study-inputs.json">Study configuration</a><a href="study-validation.json">Validation</a><a href="manifest.json">Provenance</a></nav>'
    generation_links = []
    for name, label in (('generation-complete.json', 'Completed reshower validation'), ('generation-manifest.json', 'Generator sample manifest')):
        source = operation/name
        if source.exists():
            (destination/name).write_bytes(source.read_bytes())
            generation_links.append(f'<a href="{name}">{label}</a>')
    if generation_links:
        body += '<nav>'+' · '.join(generation_links)+'</nav>'
    for selection in ordered_selections:
        body += f'<section><h2>{html.escape(selection_text(inputs["selections"][selection]))}</h2><div class="table-wrap"><table><thead><tr><th>Comparison</th>'+''.join(f'<th>{l:g} fb<sup>−1</sup></th>' for l in luminosities)+'<th>Full galleries</th></tr></thead><tbody>'
        for group in inputs['comparison_sets']:
            body += '<tr><th>'+html.escape(inputs['comparison_labels'][group])+'</th>'
            for luminosity in luminosities:
                name = f'summary-{luminosity:g}fb-folded-angle-{selection}-{group}.html'
                body += f'<td><a href="{name}">Six-panel summary</a></td>'
            body += '<td>'+ ' · '.join(f'<a href="{gallery_path(p,selection,group)}">|ηj| &lt; {eta_values[p]:g}</a>' for p in inputs['profiles'])+'</td></tr>'
        body += '</tbody></table></div></section>'
    body += '<section><h2>Retention and beam fraction</h2><p>Event-selection yields precede the pull cut. Accepted pull yields count each passing jet with half the event weight. Weighted zero/one/two-jet event counts and both luminosities are included in the <a href="retention.json">numerical export</a>.</p><div class="table-wrap"><table><thead><tr><th>|ηj| max.</th><th>Pull selection</th><th>Channel</th><th>Scenario</th><th>Jet retention</th><th>f<sub>beam</sub></th><th>Selected events, 300 fb⁻¹</th><th>Accepted pull yield, 300 fb⁻¹</th></tr></thead><tbody>'
    for profile in inputs['profiles']:
        for selection in ordered_selections:
            data = comparisons[(profile, selection, full_group)][3]
            for channel, label in CHANNELS:
                for scenario, values in data['analyses']['cutbased'][channel]['scenarios'].items():
                    efficiency = values['jet_retention_efficiency']
                    fraction = values['f_beam']
                    eff_text = 'undefined' if efficiency is None else f'{100*efficiency:.2f}%'
                    fraction_text = 'undefined' if fraction is None else f'{fraction:.5f}'
                    body += f'<tr><td>{eta_values[profile]:g}</td><td>{html.escape(selection_text(inputs["selections"][selection]))}</td><td>{label}</td><td>{html.escape(values["label"])}</td><td>{eff_text}</td><td>{fraction_text}</td><td>{300000*values["selected_cross_section_pb"]:,.1f}</td><td>{300000*values["accepted_pull_cross_section_pb"]:,.1f}</td></tr>'
    body += '</tbody></table></div></section>'
    body += '<footer>All selections use the same common particle threshold. Original reports remain available from the results index. The shared hard-generation cuts and original cross sections are documented in the <a href="../cutbased-herwig-preco-mg5-particlept-0p1-pullcut-0p0015/tables/cuts-and-cross-sections.tex">earlier LaTeX tables</a>; the study configuration records the shower scenario used here.</footer>'
    (destination/'index.html').write_text(page(report_title, body))
    write_json(destination/'retention.json', retention)
    write_json(destination/'study-inputs.json', inputs)
    write_json(destination/'study-status.json', state)
    write_json(destination/'study-validation.json', validation)
    (destination/'README.md').write_text(f'# {report_title}\n\n{report_description}.\n\n{report_note}\n\nOpen index.html for all summaries and galleries. The four selections are uncut, |t| > 0.0015, 0.003 and 0.006 per measured tagging jet. Each accepted jet contributes half the event weight. All inputs have particle pT > 0.1 GeV.\n\n{work_summary} The 24 summary pages and 36 galleries include both channels, eta 2/3/4.5 and 300/3000 fb^-1. All five pull observables, expected-data and finite-MC uncertainties, bin fractions, f_beam and shape/rate statistics are included.\n\nThe summaries hold the nominal prediction fixed for ratio errors. Generation-level m(jj) > 200 GeV and reconstructed m(jj) > 400 GeV remain distinct cuts. Original generator cross sections differ from common nominal-Herwig plotting normalization. BR(H -> gamma gamma) = 0.002270 is applied once; Z includes ee and mu mu.\n\nClick a plot to enlarge it and access embedded PNG/PDF downloads; Escape closes the viewer. Reused and newly generated run IDs, exact commands, code hashes, input configuration and validation are recorded in the JSON files.\n')
    manifest = {'analysis_version': analysis.ANALYSIS_VERSION, 'git': state['git'],
        'report_git': analysis.git_provenance(ROOT), 'code_sha256': state['code_sha256'],
        'report_code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__).resolve(), ROOT/'scripts/build_pullcut_report.py', ROOT/'scripts/pullcut_study_config.py')},
        'source_runs': sorted(sources), 'comparisons': {comparison_key(*key): value[0].name for key,value in comparisons.items()},
        'summaries': summaries, 'files': {str(p.relative_to(destination)): digest(p) for p in sorted(destination.rglob('*')) if p.is_file() and p.name != 'manifest.json'}}
    write_json(destination/'manifest.json', manifest)
    # Galleries link to the landing page, which in turn links to the manifest.
    # Check the complete artifact graph only after all targets exist.
    for path in destination.rglob('*.html'):
        analysis.validate_html_links(path)
    landing = results/'index.html'
    document = landing.read_text()
    marker = '<!-- '+destination.name+' -->'
    if marker not in document:
        relative = destination.relative_to(results).as_posix()
        link = marker+f'<section style="margin:24px 0;padding:20px;border:1px solid #c8d4e5;border-radius:10px;background:#eaf0f8"><h2>{html.escape(report_title)}</h2><p>{html.escape(report_description)}; uncut and |t| &gt; 0.0015, 0.003, 0.006; |ηj| &lt; 2, 3, 4.5; 300 and 3000 fb<sup>−1</sup>. <a href="{relative}/index.html">Open all 24 summaries and 36 galleries</a>.</p></section>'
        document = document.replace('</header>', '</header>'+link, 1) if '</header>' in document else document.replace('<body>', '<body>'+link, 1)
        assert marker in document
        analysis.atomic_write_text(landing, document)
    analysis.validate_html_links(landing)
    print(destination/'index.html')
