"""Study orchestration, scenario identity and cross-report numerical consistency."""
import copy
import json
from pathlib import Path
import unittest

import HwSimPythonAnalysis as analysis
from scripts.pullcut_study_config import validate_inputs, event_arguments, comparison_arguments
from scripts.build_pullcut_scan_report import validate_pairwise


class PullcutStudyTests(unittest.TestCase):
    def setUp(self):
        self.inputs = json.loads((Path(__file__).resolve().parents[1]/'studies/pullcut-scan-20260923.json').read_text())

    def test_configured_work_counts_and_threshold_arguments(self):
        inputs = validate_inputs(self.inputs)
        fresh = comparisons = reused = 0
        for profile in inputs['profiles']:
            for selection, threshold in inputs['selections'].items():
                for name in inputs['scenarios']:
                    old = inputs['reuse_sources'].get(profile, {}).get(selection, {}).get(name)
                    fresh += old is None
                    command = event_arguments(inputs, name, profile, selection)
                    self.assertEqual(command[command.index('--particle-pt-min')+1], '0.1')
                    if threshold is not None:
                        self.assertEqual(float(command[command.index('--pull-magnitude-min')+1]), threshold)
                    else:
                        self.assertNotIn('--pull-magnitude-min', command)
                for group in inputs['comparison_sets']:
                    old = inputs['reuse_comparisons'].get(profile, {}).get(selection, {}).get(group)
                    comparisons += old is None
                    reused += old is not None
        self.assertEqual((fresh, comparisons, reused), (18, 30, 6))

    def test_pairwise_membership_colors_and_markers(self):
        runs = {name: Path('/runs')/name for name in self.inputs['scenarios']}
        command = comparison_arguments(self.inputs, 'herwig-vs-pythia', runs)
        self.assertEqual(command[1:3], ['/runs/herwig', '/runs/mg5-pythia'])
        self.assertIn('mg5-pythia8=#8A5FBF', command)
        self.assertFalse(any('preco' in x for x in command))
        for index in (1, 2):
            self.assertEqual(analysis.comparison_marker('mg5-pythia8', index), '^')
            self.assertEqual(analysis.comparison_marker('mg5-pythia8-dipolerecoil', index), '^')
            self.assertEqual(analysis.comparison_marker('herwig-preco025', index), 's')
        self.assertEqual(analysis.comparison_marker('unknown', 1), 's')

    def test_reject_invalid_inputs(self):
        for threshold in (0, -1, float('nan'), float('inf')):
            inputs = copy.deepcopy(self.inputs)
            inputs['selections']['pullcut-0p003'] = threshold
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                validate_inputs(inputs)
        inputs = copy.deepcopy(self.inputs)
        inputs['comparison_sets']['herwig-vs-pythia'].reverse()
        with self.assertRaisesRegex(ValueError, 'nominal'):
            validate_inputs(inputs)

    def test_pairwise_closure_detects_changed_prediction(self):
        channels = {}
        for channel in ('higgs', 'z'):
            channels[channel] = {'scenarios': {k: {'f_beam': .5} for k in ('nominal','preco','pythia')},
                'observables': {'folded_pull_angle': {'bin_edges': [0,1],
                    'luminosities': {'300.0': {k: {'total_yield': [1.]} for k in ('nominal','preco','pythia')}},
                    'comparisons_to_reference': {'preco': {}, 'pythia': {}}}}}
        full = {'analyses': {'cutbased': channels}}
        pair = copy.deepcopy(full)
        for channel in pair['analyses']['cutbased'].values():
            del channel['scenarios']['preco']
            del channel['observables']['folded_pull_angle']['luminosities']['300.0']['preco']
            del channel['observables']['folded_pull_angle']['comparisons_to_reference']['preco']
        validate_pairwise(full, pair)
        pair['analyses']['cutbased']['z']['observables']['folded_pull_angle']['luminosities']['300.0']['pythia']['total_yield'][0] = 2.
        with self.assertRaises(AssertionError):
            validate_pairwise(full, pair)
