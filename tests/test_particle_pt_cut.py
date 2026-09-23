"""Writer-equivalent particle selection before all analysis reconstruction."""
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import HwSimPythonAnalysis as a


def objects(pt, pid=None, phi=None):
    pt = np.asarray(pt, dtype=float)
    phi = np.zeros_like(pt) if phi is None else np.asarray(phi)
    record = np.zeros((8, len(pt)))
    record[0] = pt
    record[1] = pt * np.cos(phi)
    record[2] = pt * np.sin(phi)
    record[4] = 211 if pid is None else pid
    return record


class ParticlePtCutTests(unittest.TestCase):
    def setUp(self):
        self.spec = a.SampleSpec('VBFH', 'higgs', ('unused.root',), 1., 'VBF H', '#275DAD', 0, 'signal')
        self.config = a.AnalysisConfig('Data', (300., 3000.), (self.spec,), 'test',
                                       particle_pt_min_gev=.1, pull_magnitude_min=.0015)

    def test_strict_boundary_order_and_input_unchanged(self):
        data = objects([0., .01, .1, np.nextafter(.1, math.inf), .2, 25.],
                       [211, 22, 13, -211, 12, -13])
        original = data.copy()
        particles = a.particles_from_root(data, 6, .1)
        np.testing.assert_array_equal(particles.pid, [-211, 12, -13])
        np.testing.assert_array_equal(particles.pt, data[1, 3:])
        np.testing.assert_array_equal(data, original)

    def test_omission_preserves_unfiltered_input_and_flat_records(self):
        data = objects([0., .03, .1, .1001, 30.])
        for record in (data, data.reshape(-1)):
            uncut = a.particles_from_root(record, 5)
            np.testing.assert_array_equal(uncut.pt, data[1])
            filtered = a.particles_from_root(record, 5, .1)
            np.testing.assert_array_equal(filtered.pt, data[1, 3:])
        self.assertEqual(len(a.particles_from_root(data, 3, .1)), 0)

    def test_filter_precedes_photon_isolation_and_reindexes_candidates(self):
        # Ninety 80-MeV particles would spoil isolation of the 65-GeV photon.
        data = objects([.08]*90 + [65., 60., .2, 5.],
                       [211]*90 + [22, 22, 211, 12], [0.]*91 + [math.pi, 0., 0.])
        old = a.particles_from_root(data, data.shape[1])
        self.assertGreater(old.relative_photon_isolation(90), .1)
        self.assertIsNone(a.select_higgs_candidate(old).candidate)
        particles = a.particles_from_root(data, data.shape[1], .1)
        decision = a.select_higgs_candidate(particles)
        self.assertIsNotNone(decision.candidate)
        self.assertEqual((decision.candidate.leading_index, decision.candidate.subleading_index), (0, 1))
        self.assertAlmostEqual(particles.relative_photon_isolation(0), .2/65.)
        self.assertAlmostEqual(decision.candidate.p4.mass, math.sqrt(4*65*60))

        class PseudoJet:
            def __init__(self, *p4): self.p4 = p4
            def set_user_index(self, index): self.index = index
        class Cluster:
            def __init__(self, inputs, _): self.inputs = inputs
            def inclusive_jets(self, _): return []
        fastjet = SimpleNamespace(PseudoJet=PseudoJet, ClusterSequence=Cluster, sorted_by_pt=lambda x:x)
        cluster, _ = a.cluster_selected_jets(particles, (0, 1), fastjet, object())
        self.assertEqual(len(cluster.inputs), 1)  # the retained hadron, not the neutrino
        self.assertAlmostEqual(cluster.inputs[0].p4[0], .2)
        self.assertEqual(cluster.inputs[0].index, 2)

    def test_event_loop_filters_before_selection_without_losing_event_weights(self):
        class Tree:
            objects = objects([.02, .1, .2])
            numparticles = 3
            evweight = 0.
            def GetEntries(self): return 2
            def SetBranchStatus(self, *_): pass
            def GetEntry(self, i): self.evweight = (2., 5.)[i]
        tree = Tree()
        file = SimpleNamespace(IsZombie=lambda:False, Get=lambda _:tree, Close=lambda:None)
        root = SimpleNamespace(TFile=SimpleNamespace(Open=lambda *_:file))
        fastjet = SimpleNamespace(antikt_algorithm=object(), JetDefinition=lambda *_:object())
        shard = a.SampleShard(0, 1, 0, 2, (a.EventRange(0, 0, 2),))
        with mock.patch.object(a, 'select_higgs_candidate', return_value=a.CandidateDecision((), None)) as select:
            result = a.analyze_sample_shard(root, fastjet, self.config, self.spec, 2, 7.,
                                            [{'path':'unused.root','entries':2,'sumw':7.}], shard)
        for call in select.call_args_list:
            np.testing.assert_array_equal(call.args[0].pt, [.2])
        self.assertEqual(result.processed_entries, 2)
        self.assertEqual(result.invalid_events, 0)
        self.assertEqual(result.generated_sumw, 7.)
        self.assertEqual(result.cutflow['all_events'].sumw, 7.)
        self.assertEqual(result.cutflow['all_events'].sumw2, 29.)

    def test_configuration_checkpoint_and_legacy_loading(self):
        result = a.initialize_result(self.spec, 1, 2.)
        result.processed_entries = 1
        result.processed_sumw = 2.
        a.fill_cut_steps(result, a.cutflow_steps('higgs'), 2.)
        pull = a.PullVector(.002, 0., .002, .002, 0., False)
        a.fill_pull_histograms(result, [pull, pull], 2., .0015)
        payload = a.resolved_config_payload(self.config, None)
        self.assertEqual(payload['particle_pt_min_gev'], .1)
        with tempfile.TemporaryDirectory() as temp:
            a.write_root_pass_checkpoint(Path(temp), payload, [result], 'particle-cut')
            config, restored, _ = a.load_root_pass_checkpoint(Path(temp))
            self.assertEqual(config.particle_pt_min_gev, .1)
            np.testing.assert_array_equal(restored[0].pull_bin_sumw, result.pull_bin_sumw)
        del payload['particle_pt_min_gev']
        self.assertIsNone(a.analysis_config_from_payload(payload).particle_pt_min_gev)

    def test_comparison_rejects_different_constituent_definitions(self):
        result = a.initialize_result(self.spec, 1, 1.)
        metadata = {'configuration': {'cuts':dict(a.CUTS)}}
        first = a.ComparisonSource(Path('/first'), self.config, (result,), metadata, a.ScenarioSpec('a','A'))
        for threshold in (None, .2):
            second = a.ComparisonSource(Path('/second'), a.replace(self.config, particle_pt_min_gev=threshold),
                                        (result,), metadata, a.ScenarioSpec('b','B'))
            with self.assertRaisesRegex(ValueError, 'input-particle'):
                a.validate_comparison_sources([first, second], ['cutbased'])

    def test_cli_requires_fresh_cutbased_run_and_valid_positive_threshold(self):
        args = a.parse_arguments(['--samples','test.json','--particle-pt-min','.1'])
        self.assertEqual(args.particle_pt_min, .1)
        for value in ('0','-1','nan','inf'):
            with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                a.parse_arguments(['--samples','test.json','--particle-pt-min',value])
            with self.assertRaises(ValueError):
                a.replace(self.config, particle_pt_min_gev=float(value))
        for args in (['--from-run','old'], ['--resume-incomplete','old'],
                     ['--compare-runs','one','two'], ['--samples','test.json','--analyses','xgboost']):
            with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                a.parse_arguments(args + ['--particle-pt-min','.1'])


if __name__ == '__main__':
    unittest.main()
