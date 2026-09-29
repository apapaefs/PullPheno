"""Event-loop, shard and archive integration for uncut beam asymmetries."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import HwSimPythonAnalysis as a
from beam_asymmetry import ASYMMETRY_KEYS, asymmetry_statistics


class BeamAsymmetryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.spec = a.SampleSpec('VBFH', 'higgs', ('unused.root',), 1.,
                                 'VBF H', '#275DAD', 0, 'signal')
        self.config = a.AnalysisConfig('Data', (300., 3000.), (self.spec,), 'test.json')
        self.events = [(2., (self.pull(0., 0.), self.pull(.001, -.001))),
                       (3., (self.pull(.003, .002), self.pull(-.004, .001))),
                       (5., (self.pull(.04, 0.), self.pull(-.05, -.04)))]

    @staticmethod
    def pull(beam, phi):
        magnitude = math.hypot(beam, phi)
        return a.PullVector(beam, phi, beam, magnitude, math.atan2(phi, beam), magnitude == 0.)

    def result(self, indices=range(3), threshold=None):
        result = a.initialize_result(self.spec, 3, 10.)
        for i in indices:
            weight, pulls = self.events[i]
            result.processed_entries += 1
            result.processed_sumw += weight
            a.fill_cut_steps(result, a.cutflow_steps('higgs'), weight)
            a.fill_pull_histograms(result, pulls, weight, threshold)
        return result

    def compare(self, first, second):
        for key in ASYMMETRY_KEYS:
            for field in ('sumw', 'event_second_sumw', 'mc_second_sumw2',
                          'positive_sumw', 'negative_sumw'):
                np.testing.assert_allclose(getattr(first.pull_asymmetry_moments[key], field),
                                           getattr(second.pull_asymmetry_moments[key], field))
            self.assertEqual(first.pull_asymmetry_moments[key].event_count,
                             second.pull_asymmetry_moments[key].event_count)

    def test_magnitude_selection_does_not_change_asymmetries(self):
        uncut, cut = self.result(), self.result(threshold=.006)
        self.assertNotEqual(uncut.pull_total_sumw, cut.pull_total_sumw)
        self.compare(uncut, cut)
        self.assertEqual(asymmetry_statistics([uncut], (300., 3000.)),
                         asymmetry_statistics([cut], (300., 3000.)))
        a.validate_results([cut], False)

    def test_shards_and_checkpoint_preserve_uncut_moments(self):
        full = self.result(threshold=.006)
        files = [{'path': 'unused.root', 'entries': 3, 'sumw': 10.}]
        shards = a.build_sample_shards(files, None, 3)
        merged = a.merge_sample_shard_results(self.spec, 3, 10., files,
                  [(shards[i], self.result([i], .006)) for i in range(3)])
        self.compare(full, merged)
        a.validate_results([merged], False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = a.replace(self.config, pull_magnitude_min=.006)
            a.write_root_pass_checkpoint(root, a.resolved_config_payload(config, None), [merged], 'test')
            _, loaded, _ = a.load_root_pass_checkpoint(root)
            self.compare(full, loaded[0])
            a.validate_results(loaded, False)

    def test_completed_archive_and_legacy_missing_moments(self):
        result = self.result()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a.write_histogram_npz(root, [result])
            (root / 'run.json').write_text(json.dumps({
                'status': 'complete', 'analysis_version': a.ANALYSIS_VERSION,
                'configuration': a.resolved_config_payload(self.config, None)}))
            (root / 'summaries' / 'analysis.json').write_text(json.dumps({
                'samples': [a.result_summary(result, self.config.luminosities_fb)]}))
            _, loaded, _ = a.load_completed_run(root)
            self.compare(result, loaded[0])
            archive = root / 'summaries' / 'histograms.npz'
            with np.load(archive) as saved:
                old = {k: saved[k].copy() for k in saved.files if '__asymmetry__' not in k}
            np.savez(archive, **old)
            metadata = json.loads((root / 'run.json').read_text())
            with self.assertRaisesRegex(ValueError, 'moments are missing'):
                a.load_completed_run(root)
            metadata['analysis_version'] = '2.9.0'
            metadata['configuration']['analysis_version'] = '2.9.0'
            metadata['configuration'].pop('beam_asymmetries')
            metadata['run_id'] = 'legacy-source'
            metadata['partial'] = False
            (root / 'run.json').write_text(json.dumps(metadata))
            _, legacy, _ = a.load_completed_run(root)
            self.assertEqual(legacy[0].pull_asymmetry_moments, {})
            with self.assertRaisesRegex(ValueError, 'reanalyse'):
                asymmetry_statistics(legacy, (300.,))
            # Replotting ordinary legacy observables must not falsely advertise
            # new moments or create an archive the same version cannot reload.
            output = root / 'derived'
            with mock.patch.object(a, 'generate_plots', return_value=[]), \
                 mock.patch.object(a, 'generate_ri_plots', return_value=[]), \
                 mock.patch.object(a, 'process_root_samples', side_effect=AssertionError('ROOT reopened')), \
                 mock.patch('sys.stdout'):
                self.assertEqual(a.main(['--from-run', str(root), '--output-root', str(output),
                                         '--run-name', 'legacy-replot']), 0)
            derived = next((output / 'runs').iterdir())
            _, reloaded, derived_metadata = a.load_completed_run(derived)
            self.assertFalse(derived_metadata['configuration']['beam_asymmetries']['available'])
            self.assertEqual(reloaded[0].pull_asymmetry_moments, {})
            self.assertIsNone(reloaded[0].beam_asymmetry_model)
            a.validate_results(reloaded, False)

    def test_missing_checkpoint_keys_and_corrupt_signs_are_rejected(self):
        result = self.result()
        result.pull_asymmetry_moments['pull_asymmetry_beam'].sumw[0] += 1.
        with self.assertRaisesRegex(RuntimeError, 'Analysis validation failed'):
            a.validate_results([result], False)
        missing = self.result()
        missing.pull_asymmetry_moments.clear()
        with self.assertRaisesRegex(RuntimeError, 'Analysis validation failed'):
            a.validate_results([missing], False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a.write_root_pass_checkpoint(root, a.resolved_config_payload(self.config, None),
                                         [self.result()], 'test')
            checkpoint = root / a.ROOT_PASS_CHECKPOINT_RELATIVE / 'checkpoint.json'
            metadata = json.loads(checkpoint.read_text())
            metadata['samples'][0].pop('pull_asymmetry_keys')
            checkpoint.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'moments are missing'):
                a.load_root_pass_checkpoint(root)


if __name__ == '__main__':
    unittest.main()
