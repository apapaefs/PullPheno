#!/usr/bin/env python3
"""Write LaTeX tables from the resolved analysis and archived campaign records."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import HwSimPythonAnalysis as analysis


def run_card(text):
    result = {}
    for line in text.splitlines():
        line = line.split('!', 1)[0].strip()
        if '=' in line and not line.startswith('#'):
            value, key = line.split('=', 1)
            result[key.strip().lower()] = value.strip()
    return result


def rounded_measurement(value, uncertainty):
    decimals = max(0, 1-int(math.floor(math.log10(uncertainty))))
    return rf'${value:.{decimals}f}\pm{uncertainty:.{decimals}f}$'


def table(caption, label, rows, columns=r'@{}>{\raggedright\arraybackslash}p{.22\textwidth}X@{}', header=None):
    result = [r'\begin{table}[p]', r'\centering\small', r'\renewcommand{\arraystretch}{1.16}',
              r'\caption{'+caption+'}', r'\label{'+label+'}',
              r'\begin{tabularx}{\textwidth}{'+columns+'}', r'\toprule']
    if header:
        result += [' & '.join(header)+r' \\', r'\midrule']
    result += [' & '.join(row)+r' \\' for row in rows]
    result += [r'\bottomrule', r'\end{tabularx}', r'\end{table}', '']
    return '\n'.join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provenance-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--particle-pt-min',type=float,
                        help='Common strict input-particle threshold applied before reconstruction, in GeV')
    args = parser.parse_args()
    if args.particle_pt_min is not None and (not math.isfinite(args.particle_pt_min) or args.particle_pt_min<=0):
        parser.error('--particle-pt-min must be finite and positive')
    provenance = json.loads((args.provenance_dir/'generation-provenance.json').read_text())
    manifest_files = [f for f in provenance['files'] if f['path'].endswith('.json')]
    manifests = [json.loads(f['text']) for f in manifest_files]
    assert len(manifests) == 3
    samples = [{s['name']:s for s in m['samples']} for m in manifests]
    for process, item in provenance['lhe_cards'].items():
        card = run_card(item['cards']['MGRunCard'])
        assert float(card['ptj']) == 20 and float(card['etaj']) == 5 and float(card['mmjj']) == 200
        assert float(card['drjj']) == 0 and int(card['lhaid']) == 13200
        if process == 'aajj':
            assert [float(card[k]) for k in ('ptgmin','etaa','draa','mmaa','mmaamax','r0gamma','xn','epsgamma')] == [20,2.7,.4,115,135,.4,1,1]
        if process in ('VBFZ','QCDZjj'):
            assert card['cut_decays'].lower() == 'false'
            assert float(card['bwcutoff']) == 15
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    c = analysis.resolved_cuts_for_profile('eta45-mll71-111')[0]
    assert c['jet_abs_eta_max'] == 4.5
    f = lambda key: f'{c[key]:g}'
    generation_rows = [
        (r'Collision energy', r'$pp$ collisions at $\sqrt{s}=13.6$ TeV.'),
        (r'PDF and coupling', r'CT14lo (LHAPDF ID 13200), $\alpha_s(M_Z)=0.118$.'),
        (r'Hard scales', r'$\mu_R=\mu_F=125$ GeV for $Hjj$ and $\gamma\gamma jj$; $\mu_R=\mu_F=91.1876$ GeV for $Zjj$.'),
        (r'Parton flavours', r'$j=g,u,d,s,c,b,\bar u,\bar d,\bar s,\bar c,\bar b$; five-flavour PDFs.'),
        (r'All five processes', r'Each final-state QCD parton: $p_{T,j}>20$ GeV and $|y_j|<5$; $m_{jj}>200$ GeV. These are hard-parton cuts before showering and jet clustering. No additional nonzero $\Delta R_{jj}$ requirement.'),
        (r'$\gamma\gamma jj$: photons', r'$p_{T,\gamma}>20$ GeV, $|y_\gamma|<2.7$, $\Delta R_{\gamma\gamma}>0.4$, and $115<m_{\gamma\gamma}<135$ GeV.'),
        (r'$\gamma\gamma jj$: isolation', r'Smooth-cone isolation with $R_0=0.4$, $n=1$, $\epsilon_\gamma=1$, applied to QCD partons. Herwig: \texttt{CutType=VBFNLO}; MG5: \texttt{isoEM=False} and \texttt{ptgmin=20} GeV.'),
        (r'VBF $Hjj$', r'Electroweak production at $\mathcal{O}(\alpha_{\rm EW}^3)$, with the VBF diagram restriction. $m_H=125$ GeV; $H\to\gamma\gamma$ is forced after hard production.'),
        (r'QCD $Hjj$', r'Loop-induced production at $\mathcal{O}(\alpha_s^4\alpha_{\rm EW})$ with $m_t=173$ GeV; $H\to\gamma\gamma$ is forced. No hard photon cuts are imposed on the Higgs decay.'),
        (r'Continuum $\gamma\gamma jj$', r'Production at $\mathcal{O}(\alpha_s^2\alpha_{\rm EW}^2)$, with the photon cuts above.'),
        (r'VBF and QCD $Zjj$', r'Production orders $\mathcal{O}(\alpha_{\rm EW}^3)$ and $\mathcal{O}(\alpha_s^2\alpha_{\rm EW})$, respectively, before decay. Both $Z\to e^+e^-$ and $Z\to\mu^+\mu^-$ are included.'),
        (r'Herwig $Z$ treatment', r'On-shell hard $Z$ production (\texttt{HardProcessWidth=0}); restricted $e,\mu$ decays with the branching-ratio reweighter. No active hard lepton acceptance or explicit dilepton mass-window cut.'),
        (r'MG5 $Z$ treatment', r'Decay-chain matrix elements with \texttt{bwcutoff=15}, i.e. the resonant mass range $M_Z\pm15\Gamma_Z$, and \texttt{cut\_decays=False}; no additional hard lepton cuts.'),
        (r'Herwig variation', r'The $p_{\rm reco}=0.25$ samples change \texttt{ReconnectionProbability} to 0.25; the hard-process generation cuts are shared with nominal Herwig.'),
    ]
    basic_rows = [
        (r'$H\to\gamma\gamma$: multiplicity', r'At least two photons; use the two leading isolated accepted photons.'),
        (r'Photon acceptance', rf'$|\eta_\gamma|<{f("photon_abs_eta_max")}$; $p_{{T,\gamma_1}}>{f("photon_pt_lead_min_gev")}$ GeV and $p_{{T,\gamma_2}}>{f("photon_pt_sublead_min_gev")}$ GeV.'),
        (r'Photon isolation', rf'$\sum_{{i\ne\gamma}}p_{{T,i}}/p_{{T,\gamma}}<{f("photon_relative_isolation_max")}$ within $\Delta R_{{\eta\phi}}<{f("photon_isolation_dr")}$, using stored visible particles and excluding the photon itself and neutrinos.'),
        (r'Diphoton mass', rf'${f("photon_mass_min_gev")}<m_{{\gamma\gamma}}<{f("photon_mass_max_gev")}$ GeV.'),
        (r'$Z\to\ell^+\ell^-$: pair', r'$\ell=e,\mu$; at least two leptons; choose the opposite-sign, same-flavour pair closest to $M_Z=91.1876$ GeV.'),
        (r'Lepton acceptance', rf'$|\eta_\ell|<{f("lepton_abs_eta_max")}$; $p_{{T,\ell_1}}>{f("lepton_pt_lead_min_gev")}$ GeV and $p_{{T,\ell_2}}>{f("lepton_pt_sublead_min_gev")}$ GeV.'),
        (r'Dilepton mass', rf'${f("lepton_mass_min_gev")}<m_{{\ell\ell}}<{f("lepton_mass_max_gev")}$ GeV for all three acceptance profiles. No additional lepton-isolation requirement.'),
        (r'Jet inputs', r'Stored final-state particles with finite four-momenta, $E>0$ and $p_T>0$, excluding neutrinos and the two chosen boson decay constituents. The input-record thresholds are given in Table~\ref{tab:pull-records}.'),
        (r'Jet reconstruction', rf'Anti-$k_T$, $R={f("jet_radius")}$; require at least two jets with $p_{{T,j}}\geq {f("jet_pt_min_gev")}$ GeV, following the FastJet inclusive-jet threshold.'),
        (r'Jet acceptance', r'Three separate analyses: $|\eta_j|<2$, $|\eta_j|<3$, or $|\eta_j|<4.5$. Apply acceptance before choosing the two highest-$p_T$ tagging jets.'),
        (r'Opposite hemispheres', r'$y_{j_1}y_{j_2}<0$.'),
        (r'VBF dijet mass', rf'$m_{{jj}}>{f("mjj_min_gev")}$ GeV.'),
        (r'VBF rapidity gap', rf'$|y_{{j_1}}-y_{{j_2}}|>{f("abs_delta_y_min")}$.'),
        (r'Boson centrality', r'$\min(y_{j_1},y_{j_2})<y_B<\max(y_{j_1},y_{j_2})$, where $B=H$ or $Z$.'),
        (r'Pull selection', r'Uncut baseline or the strict requirement $|\vec t_j|>0.0015$, independently for each tagging jet after the event selection. Each passing jet contributes $w_{\rm event}/2$; the jets are not reassigned.'),
        (r'Pull definition', r'$\vec t_j=\sum_{i\in j}(p_{T,i}/p_{T,j})|\vec r_i|\vec r_i$, with $\vec r_i=(y_i-y_j,\Delta\phi_{ij})$. Use $t_{\rm beam}=\mathrm{sgn}(y_j)t_y$ and the existing beam-signed angle; fold to $|\theta_s|\in[0,\pi]$.'),
        (r'Expected samples', r'$\mathcal{L}=300$ and $3000\,\mathrm{fb}^{-1}$. Correlated half-weight entries are retained in the data and finite-MC covariance matrices.'),
    ]
    record_rows = [
        (r'Herwig stored objects', r'The HwSim \texttt{objects} branch stores the final-state particles before its reconstructed-object acceptance cuts; no $p_T>0.1$ GeV or $|\eta|<6$ cut is applied to this branch.'),
        (r'MG5+Pythia stored objects', r'The Pythia writer stores final-state particles with $p_T>0.1$ GeV in the \texttt{objects} branch. No pseudorapidity cut is imposed on these raw objects.'),
        (r'Reconstruction settings', r'The writer settings $|\eta|<6$ and its reconstructed-jet thresholds act on writer-side reconstructed objects. This analysis reclusters the raw \texttt{objects} branch.'),
        (r'IRC interpretation', r'The positive pull-magnitude cut removes the undefined zero-vector angle. The individual-particle threshold in the Pythia input remains a separate collinear-sensitivity issue; these existing samples do not by themselves establish a fully IRC-safe all-particle implementation.'),
    ]
    if args.particle_pt_min is not None:
        threshold=f'{args.particle_pt_min:g}'
        common=(rf'For nominal Herwig, Herwig $p_{{\rm reco}}=0.25$ and MG5+Pythia, retain raw final-state particles only if $p_T>{threshold}$ GeV before boson selection, photon isolation, jet clustering and pull. No additional particle pseudorapidity cut is applied.')
        basic_rows.insert(0,(r'Common particle acceptance',common))
        basic_rows=[(label,text.replace('using stored visible particles','using visible particles passing the common input cut')) for label,text in basic_rows]
        record_rows.insert(2,(r'Common analysis input',common))
        record_rows=[(label,(r'The positive pull-magnitude cut removes the undefined zero-vector angle. The common individual-particle threshold retains a separate collinear sensitivity, even after matching the two generators; these samples do not establish a fully IRC-safe all-particle implementation.' if label==r'IRC interpretation' else text)) for label,text in record_rows]
    pieces = [
        table(r'Hard-process generation settings and cuts for the campaign-14 Herwig and campaign-20 MG5+Pythia samples. Generation cuts apply before the particle-level analysis.', 'tab:pull-generation', generation_rows),
        table(r'Basic object selection and subsequent VBF and pull requirements. The jet acceptance is in pseudorapidity, while the VBF topology and pull definition retain rapidity.', 'tab:pull-analysis', basic_rows),
        table(r'Particle storage and reconstruction cuts relevant to the input records. These are distinct from the hard-process cuts and the subsequent analysis selection.', 'tab:pull-records', record_rows),
    ]
    process_labels = {'VBFH':r'VBF $Hjj$', 'QCDHjj':r'QCD $Hjj$', 'aajj':r'$\gamma\gamma jj$', 'VBFZ':r'VBF $Zjj$', 'QCDZjj':r'QCD $Zjj$'}
    rows=[]; records=[]
    for process, label in process_labels.items():
        row=[label]
        for scenario, sample_map, source in zip(('herwig','preco025','mg5-pythia'),samples,manifest_files):
            sample=sample_map[process]
            row.append(rounded_measurement(sample['generator_cross_section_pb'],sample['generator_cross_section_unc_pb']))
            records.append({'scenario':scenario,'process':process,**sample,
                            'generator_cross_section_manifest':source['path'],
                            'generator_cross_section_fields':['generator_cross_section_pb','generator_cross_section_unc_pb'],
                            'comparison_reference_cross_section_pb':samples[0][process]['cross_section_pb']})
        rows.append(row)
    pieces.append(table(
        r'Original generator cross sections in pb after the generation cuts of Table~\ref{tab:pull-generation}, before analysis cuts. Errors are integration uncertainties. Sources are the campaign-14 Herwig integration outputs and campaign-20 MG5 LHE initialization records, as archived in the sample manifests. The $Hjj$ entries precede the $H\to\gamma\gamma$ branching fraction; the $Zjj$ entries include the combined $e^+e^-$ and $\mu^+\mu^-$ final states.',
        'tab:pull-cross-sections', rows, r'@{}lXXX@{}',
        ['Process','Herwig nominal',r'Herwig $p_{\rm reco}=0.25$','MG5+Pythia 8']))
    normalization_rows=[(label,rf'${samples[0][process]["cross_section_pb"]:.9g}$') for process,label in process_labels.items()]
    pieces.append(table(
        r'Final-state cross sections in pb used as the common normalization of all three scenarios in the comparison plots. Apply $\mathrm{BR}(H\to\gamma\gamma)=0.002270$ once to the nominal Herwig Higgs-production cross sections. No further branching factor is applied to the continuum or decay-inclusive $Z$ samples. Each scenario retains its own generated sum of weights and selection efficiency.',
        'tab:pull-normalization',normalization_rows,r'@{}lX@{}',['Process',r'$\sigma_{\rm reference}$ [pb]']))
    note = (r'For each scenario, the expected bin content is $1000\,\mathcal{L}_{\rm fb^{-1}}\,'
            r'\sigma_{\rm reference,pb}\,\sum_e w_e q_{e,i}/\sum_{\rm generated}w_e$, '
            r'where $q_{e,i}$ is one half of the number of accepted tagging jets in bin $i$ in event $e$.'
            '\n'+r'The quoted magnitude threshold defines a fixed fiducial measurement; no threshold optimization or claim of negligible hadronization corrections is made.'+'\n')
    fragment='% Requires booktabs, tabularx and amsmath. Generated from archived campaign provenance.\n\n'+'\n'.join(pieces)+note
    (output/'cuts-and-cross-sections.tex').write_text(fragment)
    # Keep the preview compact; the reusable fragment retains manuscript floats.
    body = (pieces[0] + r'\clearpage' + pieces[1] + r'\clearpage'
            + '\n'.join(pieces[2:]) + r'\small\medskip ' + note)
    body = body.replace(r'\begin{table}[p]', r'\begin{table}[H]')
    wrapper=(r'\documentclass[10pt,a4paper]{article}'+'\n'+r'\usepackage[margin=18mm]{geometry}'+'\n'+
             r'\usepackage{booktabs,tabularx,amsmath,amssymb,float}'+'\n'+r'\usepackage[T1]{fontenc}'+'\n'+
             r'\usepackage{lmodern}'+'\n'+r'\begin{document}'+'\n'+body+'\n'+r'\end{document}'+'\n')
    (output/'tables-preview.tex').write_text(wrapper)
    (output/'cross-sections.json').write_text(json.dumps(records,indent=2)+'\n')
    (output/'analysis-definition.json').write_text(json.dumps({'analysis_version':analysis.ANALYSIS_VERSION,'particle_pt_min_gev':args.particle_pt_min,'input_cut_stage':'before boson selection, isolation, jet clustering and pull','pull_magnitude_min':.0015,'pull_selection':'optional, strict, independently per tagging jet'},indent=2)+'\n')
    for name in ('generation-provenance.json','writer-provenance.json','stored-particles-check.json'):
        source=args.provenance_dir/name
        if source.exists(): shutil.copyfile(source,output/name)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.json')
              if p.name != 'provenance-manifest.json'}
    (output/'provenance-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(output/'cuts-and-cross-sections.tex')


if __name__ == '__main__': main()
