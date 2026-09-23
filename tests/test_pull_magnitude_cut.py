"""Acceptance and event-covariance tests for the conditional pull measurement."""
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import HwSimPythonAnalysis as a


class PullMagnitudeCutTests(unittest.TestCase):
    def setUp(self):
        self.spec = a.SampleSpec('VBFH', 'higgs', ('unused.root',), 1.0, 'VBF H', '#275DAD', 0, 'signal')
        self.config = a.AnalysisConfig('Data', (300., 3000.), (self.spec,), 'test.json', pull_magnitude_min=0.0015)
        self.events = [
            (2., (self.pull(0., 0.), self.pull(.0015, 2.9))),
            (3., (self.pull(.00150001, .2), self.pull(.001, 2.9))),
            (5., (self.pull(.004, .2), self.pull(.005, 2.9))),
        ]

    @staticmethod
    def pull(magnitude, angle):
        return a.PullVector(magnitude*math.cos(angle), magnitude*math.sin(angle),
                            magnitude*math.cos(angle), magnitude, angle, magnitude == 0.)

    def result(self, indices=range(3), threshold=.0015):
        result = a.initialize_result(self.spec, 3, 10.)
        for i in indices:
            weight, pulls = self.events[i]
            result.processed_entries += 1
            result.processed_sumw += weight
            a.fill_cut_steps(result, a.cutflow_steps('higgs'), weight)
            a.fill_pull_histograms(result, pulls, weight, threshold)
        return result

    def test_strict_boundary_zero_and_half_weight(self):
        result = self.result()
        self.assertEqual([result.pull_pass_counts[k].raw_count for k in range(3)], [1, 1, 1])
        self.assertEqual([result.pull_pass_counts[k].sumw for k in range(3)], [2., 3., 5.])
        for observable in a.PULL_OBSERVABLE_KEYS:
            histogram = result.histograms[observable]
            self.assertEqual(histogram.entries, 3)
            self.assertEqual(histogram.integral, 6.5)
            moments = result.pull_observable_moments[observable]
            self.assertAlmostEqual(moments.event_second_sumw.sum(), 5.75)
            self.assertAlmostEqual(moments.mc_second_sumw2.sum(), 27.25)
        self.assertEqual(result.zero_pull_jets, 0)
        self.assertEqual(a.validate_results([result], False)['status'], 'passed')

    def test_covariance_matches_independent_event_vectors_and_luminosity_scaling(self):
        result = self.result()
        # Fixed half weight: one beam entry, then a correlated beam/away pair.
        q = np.zeros((3, 6)); q[1, 0] = .5; q[2, 0] = .5; q[2, 5] = .5
        weights = np.array([2., 3., 5.])
        expected_bins = weights @ q
        expected_data = np.einsum('n,ni,nj->ij', weights, q, q)
        expected_mc = np.einsum('n,ni,nj->ij', weights**2, q, q)
        np.testing.assert_allclose(result.pull_bin_sumw, expected_bins)
        np.testing.assert_allclose(result.pull_event_second_sumw, expected_data)
        np.testing.assert_allclose(result.pull_mc_second_sumw2, expected_mc)
        first = a.total_pull_observable_statistics('higgs', 'cutbased', 'folded_pull_angle', [result], 300., {'VBFH': 1.})
        second = a.total_pull_observable_statistics('higgs', 'cutbased', 'folded_pull_angle', [result], 3000., {'VBFH': 1.})
        np.testing.assert_allclose(second['yield'], 10 * first['yield'])
        np.testing.assert_allclose(second['data_covariance'], 10 * first['data_covariance'])
        np.testing.assert_allclose(second['mc_covariance'], 100 * first['mc_covariance'])
        r1, c1 = a.normalized_binned_prediction(first['yield'], first['data_covariance'])
        r2, c2 = a.normalized_binned_prediction(second['yield'], second['data_covariance'])
        np.testing.assert_allclose(r1, r2)
        np.testing.assert_allclose(c2, c1 / 10)
        differential = a.differential_pull_statistics(
            expected_bins / 10, expected_data / 10, expected_mc / 100, (300., 3000.))
        np.testing.assert_allclose(differential['expected_statistical_covariance']['300.0'], c1)
        self.assertEqual(float(first['selected_cross_section_pb']), 1.)
        self.assertEqual(float(first['accepted_pull_cross_section_pb']), .65)

    def test_uncut_retains_zero_and_all_event_weight(self):
        result = self.result(threshold=None)
        self.assertEqual(result.pull_total_sumw, 10.)
        self.assertEqual(result.zero_pull_jets, 1)
        self.assertEqual(result.pull_pass_counts[2].raw_count, 3)
        self.assertEqual(a.validate_results([result], False)['status'], 'passed')

    def test_empty_accepted_measurement_is_valid(self):
        result = self.result(threshold=.1)
        self.assertEqual(a.validate_results([result], False)['status'], 'passed')
        summary = a.result_summary(result, (300.,))
        self.assertIsNone(summary['pull']['f_beam'])
        self.assertEqual(summary['pull']['jet_retention_efficiency'], 0.)

    def test_checkpoint_roundtrip_and_legacy_counts(self):
        result = self.result()
        payload = a.resolved_config_payload(self.config, None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a.write_root_pass_checkpoint(root, payload, [result], 'test-cut')
            config, loaded, _ = a.load_root_pass_checkpoint(root)
            self.assertEqual(config.pull_magnitude_min, .0015)
            self.assertEqual(loaded[0].pull_pass_counts, result.pull_pass_counts)
            self.assertEqual(a.validate_results(loaded, False)['status'], 'passed')
        legacy = self.result(threshold=None)
        a.restore_pull_pass_counts(legacy, None)
        self.assertEqual(legacy.pull_pass_counts[2].sumw, 10.)
        old_payload = a.resolved_config_payload(a.replace(self.config, pull_magnitude_min=None), None)
        del old_payload['pull_magnitude_min']
        self.assertIsNone(a.analysis_config_from_payload(old_payload).pull_magnitude_min)

    def test_shard_merge_preserves_pass_counts_and_second_moments(self):
        files = [{'path': 'unused.root', 'entries': 3, 'sumw': 10.}]
        shards = a.build_sample_shards(files, None, 3)
        merged = a.merge_sample_shard_results(self.spec, 3, 10., files,
                                              [(shards[i], self.result([i])) for i in range(3)])
        full = self.result()
        self.assertEqual(merged.pull_pass_counts, full.pull_pass_counts)
        for key in a.PULL_OBSERVABLE_KEYS:
            np.testing.assert_allclose(merged.pull_observable_moments[key].mc_second_sumw2,
                                       full.pull_observable_moments[key].mc_second_sumw2)
        self.assertEqual(a.validate_results([merged], False)['status'], 'passed')

    def test_parser_validation_and_eta45(self):
        args = a.parse_arguments(['--samples', 'test.json', '--pull-magnitude-min', '.0015',
                                  '--cut-profile', 'eta45-mll71-111'])
        self.assertEqual(args.pull_magnitude_min, .0015)
        for value in ('0', '-1', 'nan', 'inf'):
            with self.subTest(value=value), mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                a.parse_arguments(['--samples', 'test.json', '--pull-magnitude-min', value])
        for args in (['--from-run', 'old'], ['--resume-incomplete', 'old'],
                     ['--samples', 'test.json', '--analyses', 'xgboost']):
            with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                a.parse_arguments(args + ['--pull-magnitude-min', '.0015'])
        cuts, variable = a.resolved_cuts_for_profile('eta45-mll71-111')
        self.assertEqual(variable, 'pseudorapidity')
        self.assertEqual(cuts['jet_abs_eta_max'], 4.5)
        self.assertEqual((cuts['lepton_mass_min_gev'], cuts['lepton_mass_max_gev']), (71., 111.))

    def test_comparison_rejects_threshold_or_acceptance_mismatch(self):
        result = self.result()
        scenario = a.ScenarioSpec('nominal', 'Nominal')
        other = a.ScenarioSpec('other', 'Other')
        metadata = {'configuration': {'cuts': dict(a.CUTS)}}
        first = a.ComparisonSource(Path('/first'), self.config, (result,), metadata, scenario)
        for field, value, error in [('pull_magnitude_min', None, 'threshold'),
                                     ('jet_acceptance_variable', 'pseudorapidity', 'acceptance')]:
            second = a.ComparisonSource(Path('/second'), a.replace(self.config, **{field: value}),
                                        (result,), metadata, other)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                a.validate_comparison_sources([first, second], ['cutbased'])

    def test_pi_tick_positions_and_labels(self):
        for key, low in [('folded_pull_angle', 0.), ('signed_pull_angle', -math.pi)]:
            axis = mock.Mock()
            a.format_pull_angle_axis(axis, key)
            positions, labels = axis.set_xticks.call_args.args
            np.testing.assert_allclose(positions, np.linspace(low, math.pi, 7))
            self.assertIn(r'$\pi$', labels)
            self.assertIn(r'$0$', labels)

    def test_scan_threshold_boundaries_moments_and_recovery(self):
        for threshold in (.003, .006):
            with self.subTest(threshold=threshold):
                self.events = [
                    (2., (self.pull(0., 0.), self.pull(threshold, 2.9))),
                    (3., (self.pull(np.nextafter(threshold, math.inf), .2), self.pull(threshold*.7, 2.9))),
                    (5., (self.pull(threshold*2, .2), self.pull(threshold*3, 2.9))),
                ]
                result = self.result(threshold=threshold)
                self.assertEqual([result.pull_pass_counts[k].raw_count for k in range(3)], [1, 1, 1])
                for key in a.PULL_OBSERVABLE_KEYS:
                    histogram = result.histograms[key]
                    rows = []
                    for weight, pulls in self.events:
                        values = []
                        for pull in pulls:
                            if pull.magnitude <= threshold:
                                continue
                            values.append({'pull_magnitude': pull.magnitude, 'pull_t_beam': pull.t_beam,
                                           'pull_t_phi': pull.t_phi, 'signed_pull_angle': pull.signed_angle,
                                           'folded_pull_angle': abs(pull.signed_angle)}[key])
                        rows.append(np.histogram(values, bins=histogram.edges)[0]/2.)
                    rows = np.asarray(rows); weights = np.array([2., 3., 5.])
                    expected = weights @ rows
                    expected_data = np.einsum('n,ni,nj->ij', weights, rows, rows)
                    expected_mc = np.einsum('n,ni,nj->ij', weights**2, rows, rows)
                    np.testing.assert_array_equal(histogram.sumw, expected)
                    moment = result.pull_observable_moments[key]
                    np.testing.assert_array_equal(moment.event_second_sumw, expected_data)
                    np.testing.assert_array_equal(moment.mc_second_sumw2, expected_mc)
                config = a.replace(self.config, pull_magnitude_min=threshold, particle_pt_min_gev=.1)
                with tempfile.TemporaryDirectory() as directory:
                    a.write_root_pass_checkpoint(Path(directory), a.resolved_config_payload(config, None), [result], 'scan')
                    restored_config, restored, _ = a.load_root_pass_checkpoint(Path(directory))
                    self.assertEqual(restored_config.pull_magnitude_min, threshold)
                    self.assertEqual(restored_config.particle_pt_min_gev, .1)
                    self.assertEqual(restored[0].pull_pass_counts, result.pull_pass_counts)
                    for key in a.PULL_OBSERVABLE_KEYS:
                        np.testing.assert_array_equal(restored[0].pull_observable_moments[key].event_second_sumw,
                                                      result.pull_observable_moments[key].event_second_sumw)
                files = [{'path': 'unused.root', 'entries': 3, 'sumw': 10.}]
                shards = a.build_sample_shards(files, None, 3)
                merged = a.merge_sample_shard_results(self.spec, 3, 10., files,
                    [(shards[i], self.result([i], threshold)) for i in range(3)])
                for key in a.PULL_OBSERVABLE_KEYS:
                    np.testing.assert_array_equal(merged.pull_observable_moments[key].mc_second_sumw2,
                                                  result.pull_observable_moments[key].mc_second_sumw2)


if __name__ == '__main__':
    unittest.main()
