#!/usr/bin/env python3
"""Curate the validated cut/uncut study without editing any source run."""
import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import HwSimPythonAnalysis as analysis

PROFILES = ('eta2-mll71-111', 'eta3-mll71-111', 'eta45-mll71-111')
ETAS = (2., 3., 4.5)
SELECTIONS = ('uncut', 'pullcut-0p0015')
CHANNELS = (('higgs', 'H → γγ'), ('z', 'Z → e⁺e⁻, μ⁺μ⁻'))
CSS = '''*{box-sizing:border-box}body{max-width:1500px;margin:auto;padding:28px;background:#f7f8fa;color:#172033;font:16px/1.5 system-ui,sans-serif}h1{font-size:2rem;margin-bottom:10px}h2{font-size:1.4rem;margin:0 0 10px}h3{font-size:1.05rem;margin:0 0 10px}a{color:#275dad}nav{display:flex;flex-wrap:wrap;gap:10px 24px;margin:22px 0}section{background:#fff;border:1px solid #dce2eb;border-radius:10px;padding:22px;margin:22px 0}.figures{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0;min-width:0}img{display:block;width:100%;height:auto}figcaption,footer{font-size:.87rem;color:#556070}.note{padding:14px 18px;border-left:4px solid #275dad;background:#eaf0f8}.caveat{border-left-color:#98641b;background:#fff4df}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:.9rem}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #dce2eb}th{background:#f1f4f8}.summary-links a{display:block;padding:12px;border:1px solid #dce2eb;background:white;border-radius:6px}.muted{color:#556070}@media(max-width:850px){body{padding:12px}section{padding:14px}.figures{grid-template-columns:1fr}}@media print{body{padding:0;background:white}section{break-inside:avoid;border:0;padding:0}nav{display:none}}'''
VIEWER_CSS = '''.plot-zoom>summary{display:block;list-style:none;cursor:zoom-in}.plot-zoom>summary::-webkit-details-marker{display:none}.plot-zoom>summary:focus-visible{outline:3px solid #275dad;outline-offset:4px}.plot-zoom img{display:block;width:100%;height:auto;background:white}.plot-zoom .viewer-label.close-label{display:none}.plot-zoom .viewer-label{display:block;padding:8px 12px;background:#eaf0f8;color:#172033;border-radius:6px;font:14px/1.5 system-ui,sans-serif}.plot-zoom[open]{position:fixed;inset:0;z-index:1000;margin:0;padding:20px;background:#fff;overflow:auto;overscroll-behavior:contain}.plot-zoom[open]>summary{cursor:zoom-out}.plot-zoom[open] .open-label{display:none}.plot-zoom[open] .close-label{display:block;position:sticky;top:0;z-index:1;text-align:right}.plot-zoom[open] img{width:100%;max-width:1800px;margin:12px auto}.plot-actions{display:flex;flex-wrap:wrap;justify-content:center;gap:12px;padding:12px}.plot-actions a{display:inline-block;padding:8px 16px;border:1px solid #bfc8d5;border-radius:6px;background:#eaf0f8;color:#275dad}.plot-zoom[open] .plot-actions{position:sticky;bottom:0;background:#fff}@media print{.plot-zoom[open]{position:static;overflow:visible;padding:0}.plot-zoom .viewer-label,.plot-actions{display:none}}'''
VIEWER_SCRIPT = '''<script>document.addEventListener("keydown",function(event){if(event.key==="Escape"){document.querySelectorAll(".plot-zoom[open]").forEach(function(plot){plot.removeAttribute("open");});}});</script>'''


def inline_plot(png, pdf, label, download_name):
    """View and download embedded plots without navigating to another local file."""
    png_bytes=png.read_bytes()
    assert png_bytes.startswith(b'\x89PNG\r\n\x1a\n')
    width=int.from_bytes(png_bytes[16:20],'big')
    height=int.from_bytes(png_bytes[20:24],'big')
    image_url='data:image/png;base64,'+base64.b64encode(png_bytes).decode('ascii')
    pdf_url='data:application/pdf;base64,'+base64.b64encode(pdf.read_bytes()).decode('ascii')
    download_name=html.escape(download_name,quote=True)
    return (f'<details class="plot-zoom"><summary aria-label="Enlarge or close plot: {html.escape(label,quote=True)}">'
            '<span class="viewer-label open-label">Click to enlarge or download</span>'
            '<span class="viewer-label close-label">Close enlarged plot ×</span>'
            f'<img loading="lazy" decoding="async" width="{width}" height="{height}" src="{image_url}" alt="{html.escape(label,quote=True)}"></summary>'
            f'<div class="plot-actions"><a href="{image_url}" download="{download_name}.png">Download PNG</a>'
            f'<a href="{pdf_url}" download="{download_name}.pdf">Download PDF</a></div></details>')


def gallery_path(profile, selection, comparison=None):
    suffix='' if comparison is None else '-'+comparison
    return f'galleries/{profile}-{selection}{suffix}.html'


def write_embedded_gallery(run, destination, profile, selection, records, comparison=None):
    """Keep the archived gallery intact and provide a self-contained display copy."""
    output=destination/gallery_path(profile,selection,comparison)
    output.parent.mkdir(exist_ok=True)
    document=(run/'index.html').read_text()
    configuration=json.loads((run/'run.json').read_text())['configuration']
    threshold=configuration.get('particle_pt_min_gev')
    prefix='' if threshold is None else f'particlept-{threshold:g}-'.replace('.','p')
    for record in records:
        png_href=html.escape(record['png'],quote=True)
        pdf_href=html.escape(record['pdf'],quote=True)
        name=prefix+f'{profile}-{selection}-'+('' if comparison is None else comparison+'-')+record['png'].removeprefix('plots/comparison/cutbased/').replace('/','-').removesuffix('.png')
        label=f'{record["title"]}; {record["channel"]}; {record["luminosity_fb"]} fb−1'
        viewer=inline_plot(run/record['png'],run/record['pdf'],label,name)
        pattern=r'<a class="thumb-link" href="'+re.escape(png_href)+r'">.*?</a>'
        document,count=re.subn(pattern,lambda match:viewer,document,count=1,flags=re.DOTALL)
        assert count==1, record['png']
        old=f'<p><a href="{png_href}">PNG</a> <a href="{pdf_href}">PDF</a></p>'
        assert document.count(old)==1, record['png']
        document=document.replace(old,'<p>Enlarge the plot for PNG and PDF downloads.</p>',1)

    def relocate_link(match):
        href=html.unescape(match.group(1))
        if href.startswith(('#','data:','http:','https:','mailto:')):
            return match.group(0)
        path,separator,fragment=href.partition('#')
        target=os.path.relpath((run/path).resolve(),output.parent)
        return 'href="'+html.escape(target+separator+fragment,quote=True)+'"'

    document=re.sub(r'href="([^"]*)"',relocate_link,document)
    document=document.replace('</head>',f'<style>{VIEWER_CSS}</style></head>',1)
    eta=dict(zip(PROFILES,ETAS))[profile]
    pull_min=configuration.get('pull_magnitude_min')
    selection_text='No pull-magnitude cut' if pull_min is None else f'|t| &gt; {pull_min:g} per measured jet'
    luminosity_text=' and '.join(f'{x:g}' for x in configuration['luminosities_fb'])
    document=document.replace('<main>','<main><section class="panel"><a href="../index.html">Study overview</a>'
                              f'<h2>|ηj| &lt; {eta:g} · {selection_text}</h2><p>71 &lt; m<sub>ll</sub> &lt; 111 GeV; {luminosity_text} fb<sup>−1</sup>.</p>'
                              '<p>Click a plot to enlarge it here and download its PNG or PDF. '
                              'Click again, or press Escape, to close. Images and downloads are embedded.</p></section>',1)
    document=document.replace('</body>',VIEWER_SCRIPT+'</body>',1)
    output.write_text(document)


def page(title, body):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{CSS}{VIEWER_CSS}</style></head><body>'
            f'{body}{VIEWER_SCRIPT}</body></html>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status',type=Path,required=True)
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--preserved-hashes',type=Path,
                        help='Original-run hashes; defaults to preserved-files.json beside the batch status')
    args=parser.parse_args()
    state=json.loads(args.status.read_text()); assert state['status']=='complete'
    if state.get('format_version') == 2:
        from scripts.build_pullcut_scan_report import build
        build(args, state)
        return
    for name, digest in state['code_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    results=args.results.resolve(); destination=args.destination.resolve(); destination.mkdir(exist_ok=True)
    preservation_path=args.preserved_hashes or args.status.with_name('preserved-files.json')
    preserved=json.loads(preservation_path.read_text())
    changed=[name for name,digest in preserved.items()
             if not (results/name).is_file() or hashlib.sha256((results/name).read_bytes()).hexdigest()!=digest]
    assert not changed, f'Original outputs changed: {changed}'
    preservation={'status':'passed','files_checked':len(preserved),'changed':[]}
    unit_test_path=args.status.with_name('unit-tests.json')
    unit_tests=json.loads(unit_test_path.read_text())
    assert unit_tests['status']=='passed'
    unit_digest=unit_tests.get('analysis_sha256',unit_tests['code_sha256'])
    assert unit_digest==state['code_sha256']['HwSimPythonAnalysis.py']
    resume_validation=json.loads(args.status.with_name('resume-validation.json').read_text())
    assert resume_validation['status']=='passed' and not resume_validation['root_input_reread']
    particle_pt_min=state.get('particle_pt_min_gev')
    study_tag=state.get('study_tag','v28')
    particle_text=(f'Common input-particle pT > {particle_pt_min:g} GeV, applied before boson selection, isolation and jet clustering.'
                   if particle_pt_min is not None else '')
    sources={}; comparisons={}; retention=[]; checks=[]; particle_impact=[]
    for selection in SELECTIONS:
        for profile in PROFILES:
            label=f'c14-c20-{profile}-{selection}-comparison-{study_tag}'
            run=results/'runs'/Path(state['comparisons'][label]).name
            metadata=json.loads((run/'run.json').read_text())
            config=metadata['configuration']; assert metadata['status']=='complete' and not metadata['partial']
            assert config['cut_profile']==profile and config['jet_acceptance_variable']=='pseudorapidity'
            assert config.get('pull_magnitude_min') == (None if selection=='uncut' else .0015)
            assert config.get('particle_pt_min_gev') == particle_pt_min
            records=json.loads((run/'summaries/plots.json').read_text())
            comparison=json.loads((run/'summaries/comparison.json').read_text())
            analysis.validate_html_links(run/'index.html')
            comparisons[(selection,profile)]=(run,metadata,records,comparison)
            for item in metadata['source_runs']:
                source=results/'runs'/item['run_id']
                source_config, sample_results, source_metadata=analysis.load_completed_run(source)
                analysis.validate_results(sample_results,False)
                assert not source_metadata['partial']
                assert source_config.pull_magnitude_min == config.get('pull_magnitude_min')
                assert source_config.particle_pt_min_gev == particle_pt_min
                assert source_config.jet_acceptance_variable == 'pseudorapidity'
                sources[item['run_id']]={'metadata':source_metadata,'results':sample_results}
                if particle_pt_min is not None:
                    scenario=item['scenario']['identifier']
                    name={'herwig-default':'herwig','herwig-preco025':'preco025','mg5-pythia8':'mg5-pythia'}[scenario]
                    baseline_id=state['inputs']['baseline_sources'][profile][selection][name]
                    old_config,old_results,_=analysis.load_completed_run(results/'runs'/baseline_id)
                    assert old_config.particle_pt_min_gev is None
                    assert old_config.pull_magnitude_min==source_config.pull_magnitude_min
                    old_by_name={r.spec.name:r for r in old_results}
                    for result in sample_results:
                        old=old_by_name[result.spec.name]
                        assert result.processed_entries==old.processed_entries
                        np.testing.assert_allclose(result.generated_sumw,old.generated_sumw,rtol=1e-12)
                        if name=='mg5-pythia':
                            for key,stat in result.cutflow.items():
                                assert stat.raw_count==old.cutflow[key].raw_count
                                np.testing.assert_allclose([stat.sumw,stat.sumw2],[old.cutflow[key].sumw,old.cutflow[key].sumw2],rtol=1e-9,atol=1e-7)
                            for key,hist in result.histograms.items():
                                np.testing.assert_allclose(hist.sumw,old.histograms[key].sumw,rtol=1e-9,atol=1e-7)
                                np.testing.assert_allclose(hist.sumw2,old.histograms[key].sumw2,rtol=1e-9,atol=1e-7)
                            for key,moment in result.pull_observable_moments.items():
                                for field in ('bin_sumw','event_second_sumw','mc_second_sumw2'):
                                    np.testing.assert_allclose(getattr(moment,field),getattr(old.pull_observable_moments[key],field),rtol=1e-9,atol=1e-7)
                    for channel,_ in CHANNELS:
                        predictions=[]
                        for collection in (old_results,sample_results):
                            selected=sum(config['reference_cross_sections_pb'][r.spec.name]*r.cutflow['boson_centrality'].sumw/r.generated_sumw for r in collection if r.spec.channel==channel)
                            stats=analysis.total_pull_observable_statistics(channel,'cutbased','folded_pull_angle',collection,300.,config['reference_cross_sections_pb'])
                            total=float(np.sum(stats['yield']))
                            predictions.append({'selected_events_300fb':300000*selected,'accepted_pull_yield_300fb':total,'f_beam':float(np.sum(stats['yield'][:3])/total)})
                        old,new=predictions
                        particle_impact.append({'profile':profile,'selection':selection,'scenario':scenario,'channel':channel,
                            'old_particle_pt_min_gev':None,'new_particle_pt_min_gev':particle_pt_min,
                            'baseline_run':baseline_id,'new_run':source.name,'old':old,'new':new,
                            'selected_event_change_percent':100*(new['selected_events_300fb']/old['selected_events_300fb']-1),
                            'accepted_pull_change_percent':100*(new['accepted_pull_yield_300fb']/old['accepted_pull_yield_300fb']-1),
                            'delta_f_beam':new['f_beam']-old['f_beam']})
                for result in sample_results:
                    assert result.processed_entries == result.total_entries and result.invalid_events == 0
                    if selection != 'uncut':
                        magnitude=result.histograms['pull_magnitude']
                        assert magnitude.edges[1] == .0015 and magnitude.sumw[0] == 0
                        assert result.zero_pull_jets == 0
                    entry_sumw2=sum(k*s.sumw2/4 for k,s in result.pull_pass_counts.items())
                    for observable in analysis.PULL_OBSERVABLE_KEYS:
                        histogram=result.histograms[observable]
                        np.testing.assert_allclose(histogram.integral,result.pull_total_sumw,rtol=1e-10)
                        np.testing.assert_allclose(histogram.sumw2.sum(),entry_sumw2,rtol=1e-10)
                    reference=config['reference_cross_sections_pb'][result.spec.name]
                    factor=reference/result.generated_sumw
                    stat=analysis.pull_selection_summary(result)
                    selected=result.cutflow['boson_centrality']
                    retention.append({'selection':selection,'profile':profile,
                        'particle_pt_min_gev':particle_pt_min,'pull_magnitude_min':config.get('pull_magnitude_min'),
                        'scenario':item['scenario']['identifier'],'sample':result.spec.name,'channel':result.spec.channel,
                        'source_run':source.name,'reference_cross_section_pb':reference,
                        'selected_events_raw':selected.raw_count,'selected_cross_section_pb':factor*selected.sumw,
                        'accepted_pull_cross_section_pb':factor*result.pull_total_sumw,**stat,
                        'expected_events_by_accepted_jet_count':{
                            str(l):{k:1000*l*factor*v['sumw'] for k,v in stat['pass_counts'].items()}
                            for l in (300,3000)},
                        'expected_selected_events':{str(l):1000*l*factor*selected.sumw for l in (300,3000)},
                        'expected_accepted_pull_entries_half_weight':{str(l):1000*l*factor*result.pull_total_sumw for l in (300,3000)}})
            # Check luminosity scaling against the actual exported binned predictions.
            for channel,_ in CHANNELS:
                for observable, item in comparison['analyses']['cutbased'][channel]['observables'].items():
                    for scenario,v in item['luminosities']['300.0'].items():
                        high=item['luminosities']['3000.0'][scenario]
                        np.testing.assert_allclose(high['total_yield'],10*np.asarray(v['total_yield']),rtol=2e-12,atol=1e-10)
                        np.testing.assert_allclose(high['data_statistical_covariance'],10*np.asarray(v['data_statistical_covariance']),rtol=2e-12,atol=1e-10)
                        np.testing.assert_allclose(high['mc_statistical_covariance'],100*np.asarray(v['mc_statistical_covariance']),rtol=2e-12,atol=1e-10)
            for item in metadata['source_runs']:
                sample_results=sources[item['run_id']]['results']
                for channel,_ in CHANNELS:
                    for observable in analysis.PULL_OBSERVABLE_KEYS:
                        low=analysis.total_pull_observable_statistics(channel,'cutbased',observable,sample_results,300.,config['reference_cross_sections_pb'])
                        high=analysis.total_pull_observable_statistics(channel,'cutbased',observable,sample_results,3000.,config['reference_cross_sections_pb'])
                        np.testing.assert_allclose(high['yield'],10*low['yield'],rtol=2e-12)
                        np.testing.assert_allclose(high['data_covariance'],10*low['data_covariance'],rtol=2e-12)
                        np.testing.assert_allclose(high['mc_covariance'],100*low['mc_covariance'],rtol=2e-12)
                        r1,c1=analysis.normalized_binned_prediction(low['yield'],low['data_covariance'])
                        r2,c2=analysis.normalized_binned_prediction(high['yield'],high['data_covariance'])
                        np.testing.assert_allclose(r1,r2,rtol=2e-12,atol=1e-16)
                        np.testing.assert_allclose(c2,c1/10,rtol=2e-10,atol=1e-18)
            checks.append({'profile':profile,'selection':selection,'comparison':run.name,'validation':'passed'})
    assert len(comparisons) == 6 and len(sources) == 18
    for (selection,profile),(run,metadata,records,comparison) in comparisons.items():
        write_embedded_gallery(run,destination,profile,selection,records)
    summaries=[]
    for selection in SELECTIONS:
        selection_title='No pull-magnitude cut' if selection=='uncut' else '|t| > 0.0015 per measured jet'
        for luminosity in (300,3000):
            sections=[]; exported=[]
            for eta,profile in zip(ETAS,PROFILES):
                run,metadata,records,comparison=comparisons[(selection,profile)]
                figures=[]
                for channel,label in CHANNELS:
                    matches=[r for r in records if r['kind']=='reference-stack-plus-total-ratio-data-stat' and r['channel']==channel and r['luminosity_fb']==luminosity]
                    assert len(matches)==1
                    r=matches[0]; assert r['observable']=='folded_pull_angle' and r['ratio_reference_uncertainty']=='fixed'
                    png=run/r['png']; pdf=run/r['pdf']; assert png.is_file() and pdf.is_file()
                    viewer=inline_plot(png,pdf,f'{label}; |ηj| < {eta:g}; {selection}; {luminosity} fb−1',
                                       ('' if particle_pt_min is None else f'particlept-{particle_pt_min:g}-'.replace('.','p'))+f'{profile}-{channel}-{luminosity}fb-{selection}-folded-angle')
                    figures.append(f'<figure><h3>{label}</h3>{viewer}<figcaption>{luminosity} fb<sup>−1</sup> · Expected data statistics</figcaption></figure>')
                    exported.append({'profile':profile,'channel':channel,'comparison_run':run.name,'plot':r,
                        'predictions':comparison['analyses']['cutbased'][channel]['observables']['folded_pull_angle']['luminosities'][str(float(luminosity))]})
                sections.append(f'<section id="{profile}"><h2>|η<sub>j</sub>| &lt; {eta:g}</h2><p>{selection_title}; 71 &lt; m<sub>ℓℓ</sub> &lt; 111 GeV. <a href="{gallery_path(profile,selection)}">Full comparison gallery and statistical tables</a>.</p><div class="figures">'+''.join(figures)+'</div></section>')
            name=f'summary-{luminosity}fb-folded-angle-{selection}'
            title=f'{luminosity} fb⁻¹ · Folded pull angle · {selection_title}'
            if particle_pt_min is not None: title+=f' · particle pT > {particle_pt_min:g} GeV'
            body=f'<h1>{luminosity} fb<sup>−1</sup> · Folded pull-angle comparisons</h1><p>{selection_title} · Cut-based particle-level analysis · H → γγ and Z → e⁺e⁻, μ⁺μ⁻.</p><nav><a href="index.html">Study overview</a>'
            body+=''.join(f'<a href="#{p}">|ηj| &lt; {eta:g}</a>' for p,eta in zip(PROFILES,ETAS))
            body+=f'<a href="{name}.json">Binned predictions</a></nav>'
            body+='<p class="note">Nominal Herwig processes are stacked; Herwig pReco = 0.25 is shown as orange squares and MG5+Pythia 8 as purple triangles. Each accepted tagging jet contributes half the event weight. Expected-data errors include within-event correlations. Ratio errors hold the nominal prediction fixed; the nominal band shows its own counting uncertainty. These bars contain no finite-MC or systematic uncertainty. Click any plot to enlarge it in this page and download its PNG or PDF. Click it again, or press Escape, to close. All plots and downloads are embedded for offline use.</p>'
            if particle_text:
                body+=f'<p class="note">{particle_text}</p>'
            body+=''.join(sections)+'<footer>All three comparisons use pseudorapidity acceptance and common nominal-Herwig input cross sections, with scenario-specific efficiencies. The input-particle definition and generator settings are documented in the <a href="tables/cuts-and-cross-sections.tex">LaTeX tables</a>.</footer>'
            (destination/(name+'.html')).write_text(page(title,body))
            (destination/(name+'.json')).write_text(json.dumps({'luminosity_fb':luminosity,'selection':selection,'pull_magnitude_min':None if selection=='uncut' else .0015,'particle_pt_min_gev':particle_pt_min,'plots':exported},indent=2)+'\n')
            summaries.append({'name':name,'luminosity':luminosity,'selection':selection})
    rows=[]; gallery=[]; multiplicity_rows=[]
    for eta,profile in zip(ETAS,PROFILES):
        gallery.append('<tr><th>|ηj| &lt; '+f'{eta:g}</th>'+''.join(f'<td><a href="{gallery_path(profile,selection)}">'+('Uncut' if selection=='uncut' else '|t| > 0.0015')+'</a></td>' for selection in SELECTIONS)+'</tr>')
        for channel,label in CHANNELS:
            old=comparisons[('uncut',profile)][3]['analyses']['cutbased'][channel]['scenarios']
            new=comparisons[('pullcut-0p0015',profile)][3]['analyses']['cutbased'][channel]['scenarios']
            for scenario,values in new.items():
                rows.append(f'<tr><td>{eta:g}</td><td>{label}</td><td>{html.escape(values["label"])}</td><td>{100*values["jet_retention_efficiency"]:.2f}%</td><td>{old[scenario]["f_beam"]:.5f}</td><td>{values["f_beam"]:.5f}</td><td>{300000*values["selected_cross_section_pb"]:,.1f}</td><td>{300000*values["accepted_pull_cross_section_pb"]:,.1f}</td></tr>')
                components=[r for r in retention if r['selection']=='pullcut-0p0015'
                            and r['profile']==profile and r['channel']==channel and r['scenario']==scenario]
                nk=[sum(r['expected_events_by_accepted_jet_count']['300'][str(k)] for r in components)
                    for k in range(3)]
                np.testing.assert_allclose(sum(nk),300000*values['selected_cross_section_pb'],rtol=1e-10)
                np.testing.assert_allclose(nk[1]/2+nk[2],300000*values['accepted_pull_cross_section_pb'],rtol=1e-10)
                multiplicity_rows.append(f'<tr><td>{eta:g}</td><td>{label}</td><td>{html.escape(values["label"])}</td>'
                                         +''.join(f'<td>{n:,.1f}</td>' for n in nk)+'</tr>')
    body='<h1>Pull-magnitude selection: three-sample comparison</h1><p>Nominal Herwig, Herwig pReco = 0.25, and MG5+Pythia 8 · H → γγ and Z → e⁺e⁻, μ⁺μ⁻ · 13.6 TeV.</p><p>The strict cut |t| &gt; 0.0015 is applied separately to each measured tagging jet. All profiles use |ηj| acceptance and 71 &lt; m<sub>ll</sub> &lt; 111 GeV.</p><nav><a href="../index.html">All results</a><a href="../cutbased-herwig-preco-mg5/summary-300fb-folded-angle.html">Preserved earlier summary</a><a href="tables/cuts-and-cross-sections.tex">LaTeX tables</a><a href="tables/tables-preview.pdf">Compiled tables</a><a href="retention.json">Retention and yields (JSON)</a></nav>'
    body+='<section><h2>Folded-angle summaries</h2><div class="figures summary-links">'+''.join(f'<a href="{s["name"]}.html">{s["luminosity"]} fb<sup>−1</sup> · '+('No magnitude cut' if s['selection']=='uncut' else '|t| > 0.0015')+'</a>' for s in summaries)+'</div></section>'
    if particle_pt_min is not None:
        body=body.replace('<a href="retention.json">',
            '<a href="../cutbased-herwig-preco-mg5-pullcut-0p0015/index.html">Previous particle-input study</a><a href="retention.json">',1)
        body+=f'<p class="note"><strong>Matched input-particle definition.</strong> {particle_text} All three scenarios were reprocessed from ROOT with this definition. Original raw records and earlier reports are preserved.</p>'
        body+='<p class="note caveat">The positive pull cut removes the undefined zero-vector angle. The common individual-particle threshold retains a separate collinear sensitivity; matching the threshold does not establish a fully IRC-safe all-particle measurement.</p>'
    else:
        body+='<p class="note caveat"><strong>Input-particle definition.</strong> The existing MG5+Pythia ROOT records retain particles only above pT = 0.1 GeV; Herwig records retain softer particles. The positive pull cut removes the undefined zero-vector angle, but does not undo this input threshold. The tables document this remaining issue for the IRC-safety interpretation.</p>'
    body+='<section><h2>Complete comparison galleries</h2><p>Both luminosities; all five pull observables; projected-data and finite-MC errors; correlated shape and rate tests.</p><table><thead><tr><th>Jet acceptance</th><th>No magnitude cut</th><th>Magnitude cut</th></tr></thead><tbody>'+''.join(gallery)+'</tbody></table></section>'
    body+='<section><h2>Retention and beam fraction</h2><p>Retention is the weighted fraction of tagging-jet entries passing the magnitude cut. Event-selection yields and accepted half-weight jet yields are listed separately at 300 fb<sup>−1</sup>; multiply both by ten for 3000 fb<sup>−1</sup>.</p><div class="table-wrap"><table><thead><tr><th>|ηj| max.</th><th>Channel</th><th>Scenario</th><th>Retention</th><th>f<sub>beam</sub> uncut</th><th>f<sub>beam</sub> cut</th><th>Selected events, 300 fb⁻¹</th><th>Accepted jet entries / 2, 300 fb⁻¹</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></section>'
    body+='<section><h2>Events with zero, one, or two accepted jets</h2><p>Expected event yields at 300 fb<sup>−1</sup>, after event selection and classified by the per-jet pull cut. Their sum is the event-selection yield; N<sub>1</sub>/2 + N<sub>2</sub> is the accepted half-weight jet yield. Multiply by ten for 3000 fb<sup>−1</sup>. Raw counts, weight sums, and squared-weight sums for each process are available in <a href="retention.json">retention.json</a>.</p><div class="table-wrap"><table><thead><tr><th>|ηj| max.</th><th>Channel</th><th>Scenario</th><th>N<sub>0</sub></th><th>N<sub>1</sub></th><th>N<sub>2</sub></th></tr></thead><tbody>'+''.join(multiplicity_rows)+'</tbody></table></div></section>'
    impact={name:job['baseline_comparison'] for name,job in state['jobs'].items() if 'baseline_comparison' in job}
    if impact:
        (destination/'particle-threshold-impact.json').write_text(json.dumps(impact,indent=2)+'\n')
        (destination/'particle-threshold-channel-impact.json').write_text(json.dumps(particle_impact,indent=2)+'\n')
        impact_rows=[]
        for record in particle_impact:
            if record['selection']=='pullcut-0p0015':
                eta=dict(zip(PROFILES,ETAS))[record['profile']]
                impact_rows.append(f'<tr><td>{eta:g}</td><td>{dict(CHANNELS)[record["channel"]]}</td><td>{html.escape(record["scenario"])}</td><td>{record["selected_event_change_percent"]:+.3f}%</td><td>{record["accepted_pull_change_percent"]:+.3f}%</td><td>{record["delta_f_beam"]:+.5f}</td></tr>')
        body+='<section><h2>Effect of matching the particle threshold</h2><p>Changes relative to the earlier input definition, with |t| &gt; 0.0015. The Pythia prediction already uses the particle threshold. These comparisons use the same input events, so no independent-sample errors are assigned to the shifts. Full cut and uncut <a href="particle-threshold-channel-impact.json">channel results</a> and <a href="particle-threshold-impact.json">process weight sums</a> are available.</p><div class="table-wrap"><table><thead><tr><th>|ηj| max.</th><th>Channel</th><th>Scenario</th><th>Selected-event change</th><th>Accepted-pull change</th><th>Δf<sub>beam</sub></th></tr></thead><tbody>'+''.join(impact_rows)+'</tbody></table></div></section>'
    body+='<footer>Cross sections use the common nominal-Herwig reference normalization. Generator cross sections and integration errors are reported separately in the LaTeX tables. Statistical comparisons do not include experimental or modelling systematic uncertainties. <a href="study-validation.json">Validation</a> · <a href="manifest.json">Provenance manifest</a>.</footer>'
    overview_title='Pull-magnitude cut: comparisons and tables'
    if particle_pt_min is not None: overview_title+=f' · particle pT > {particle_pt_min:g} GeV'
    (destination/'index.html').write_text(page(overview_title,body))
    (destination/'retention.json').write_text(json.dumps(retention,indent=2)+'\n')
    (destination/'study-validation.json').write_text(json.dumps({'status':'passed','comparisons':checks,'source_runs':list(sources),'luminosity_scaling':'verified yields x10, data covariance x10, MC covariance x100, shapes invariant','preservation':preservation,'unit_tests':unit_tests,'resume_cli':resume_validation,'code_sha256':state['code_sha256']},indent=2)+'\n')
    (destination/'study-status.json').write_text(json.dumps(state,indent=2)+'\n')
    (destination/'unit-tests.log').write_text(args.status.with_name('unit-tests.log').read_text())
    readme='''# Pull-magnitude selection study

Open `index.html` for the four folded-angle summaries, full comparison galleries,
retention, and manuscript tables. All three acceptances use pseudorapidity:
|eta_j| < 2, 3, and 4.5. The former |y_j| < 4.5 results remain in the earlier bundle.
Click a summary or gallery plot to enlarge it within the page; click again or press Escape
to close. The enlarged view includes embedded PNG and PDF downloads, so it does
not need to open a separate local image file.

The strict |t| > 0.0015 condition applies independently to each selected tagging jet.
Each accepted jet retains half the event weight; event-selection yields are unchanged.
The mask is shared by the magnitude, two components, signed angle, and folded angle.
Event moments preserve the correlation between the two entries. For k accepted jets,
the total contributions are w*k/2 (first moment), w*(k/2)^2 (expected-data second
moment), and w^2*(k/2)^2 (finite-MC second moment).

The six-panel summaries use common nominal-Herwig final-state cross sections and
expected-data counting errors at 300 and 3000 fb^-1. Alternative scenario/reference
ratio errors hold the nominal prediction fixed. The nominal band displays its own
counting error; finite-MC errors and correlated shape/rate tests are in the full galleries.

`tables/cuts-and-cross-sections.tex` is a reusable LaTeX fragment requiring booktabs,
tabularx, and amsmath. `tables/tables-preview.pdf` is the compiled three-page preview.
The tables distinguish original generator cross sections from the shared plotting
normalization. BR(H -> gamma gamma) = 0.002270 is applied once; Z samples already
include both electron and muon decays. Full-precision cross sections and archived
generation cards, writer definitions, source paths, and hashes accompany the tables.

## Input-particle qualification

The existing Pythia ROOT writer keeps particles only above pT = 0.1 GeV; the Herwig
raw objects retain softer particles. The positive pull cut removes the undefined
zero-vector angle, but does not remove the collinear sensitivity of this constituent
threshold. These inputs alone do not establish a fully IRC-safe all-particle measurement.

## Validation and provenance

`study-status.json` records all production commands and immutable run paths.
`study-validation.json` records the numerical and preservation checks, and
`manifest.json` hashes this bundle. There are nine fresh cut runs, three fresh
uncut eta45 runs, six derived uncut eta2/eta3 runs, and six three-scenario comparisons.
Both luminosities are calculated from the same stored moments. The production code
was frozen before processing; its SHA-256 hashes are recorded in the status file.
The original uncut runs and earlier report were checked against their saved hashes.
'''
    if particle_pt_min is not None:
        readme=readme.replace('The existing Pythia ROOT writer keeps particles only above pT = 0.1 GeV; the Herwig\nraw objects retain softer particles.',
            f'All three scenarios use the strict input-particle requirement pT > {particle_pt_min:g} GeV\nbefore boson selection, photon isolation, jet clustering and pull. This matches the\nexisting Pythia writer threshold. Herwig raw records remain unchanged.')
        readme=readme.replace('There are nine fresh cut runs, three fresh\nuncut eta45 runs, six derived uncut eta2/eta3 runs, and six three-scenario comparisons.',
            'There are eighteen fresh ROOT analyses (nine cut and nine uncut) and six\nthree-scenario comparisons. No older histogram data were reused for this definition.')
    (destination/'README.md').write_text(readme)
    manifest={'analysis_version':analysis.ANALYSIS_VERSION,'particle_pt_min_gev':particle_pt_min,'batch':str(args.status),'source_runs':list(sources),'comparisons':[c['comparison'] for c in checks], 'files':{}}
    for p in sorted(destination.rglob('*')):
        if p.is_file() and p.name != 'manifest.json' and (p.name=='unit-tests.log' or p.suffix not in ('.aux','.log','.fls','.fdb_latexmk')):
            manifest['files'][str(p.relative_to(destination))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for p in destination.rglob('*.html'): analysis.validate_html_links(p)
    # Add a link while preserving every existing curated link on the landing page.
    landing = results/'index.html'
    document = landing.read_text()
    marker = '<!-- particlept-0p1-pullcut-0p0015-study -->' if particle_pt_min is not None else '<!-- pullcut-0p0015-study -->'
    if marker not in document:
        relative = destination.relative_to(results).as_posix()
        link = (marker + '<section style="margin:24px 0;padding:20px;border:1px solid #c8d4e5;border-radius:10px;background:#eaf0f8">'
                '<h2>Pull-magnitude cut: three-sample comparisons</h2>'
                + (f'<p>{particle_text}</p>' if particle_text else '') +
                '<p>|t| &gt; 0.0015 and uncut results; |ηj| &lt; 2, 3, 4.5; 300 and 3000 fb<sup>−1</sup>; '
                f'<a href="{relative}/index.html">summaries, full galleries and LaTeX tables</a>.</p></section>')
        document = document.replace('</header>', '</header>'+link, 1) if '</header>' in document else document.replace('<body>', '<body>'+link, 1)
        assert marker in document
        analysis.atomic_write_text(landing, document)
    analysis.validate_html_links(landing)
    print(destination/'index.html')


if __name__ == '__main__': main()
