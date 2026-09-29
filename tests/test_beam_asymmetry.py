import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from beam_asymmetry import (
    ASYMMETRY_EDGES,
    ASYMMETRY_KEYS,
    BeamAsymmetryMoments,
    asymmetry_statistics,
    ratio_statistics,
)


def sample(name, events, sigma=1.0, generated_sumw=None, edges=ASYMMETRY_EDGES):
    moments = {}
    for key in ASYMMETRY_KEYS:
        accumulator = BeamAsymmetryMoments(edges)
        for values, weight in events:
            accumulator.fill(values, weight)
        moments[key] = accumulator
    return SimpleNamespace(
        spec=SimpleNamespace(name=name, cross_section_pb=sigma),
        generated_sumw=(sum(weight for _, weight in events)
                        if generated_sumw is None else generated_sumw),
        pull_asymmetry_moments=moments,
    )


class BeamAsymmetryMomentTests(unittest.TestCase):
    def test_strict_signs_magnitude_factor_and_two_half_weight_jets(self):
        result = sample("a", [((0.00025, -0.00075), 2.0)])
        moments = result.pull_asymmetry_moments[ASYMMETRY_KEYS[0]]
        self.assertEqual(len(moments.edges), 61)
        self.assertEqual(moments.sumw.shape, (62,))
        np.testing.assert_array_equal(moments.sumw[:2], [1.0, -1.0])
        self.assertEqual(moments.sumw[-1], 2.0)
        self.assertEqual(moments.event_count, 1)
        statistics = asymmetry_statistics([result], [300.0])[ASYMMETRY_KEYS[0]]
        np.testing.assert_allclose(statistics["A"][:2], [0.25, -0.75])
        # A deterministic event vector has no normalized shape uncertainty.
        np.testing.assert_allclose(statistics["mc_covariance"], 0.0, atol=1e-14)

    def test_exact_zero_and_overflow_are_in_selected_normalization(self):
        result = sample("a", [((0.03, -0.05), 1.0), ((0.0, 0.03), 3.0)])
        moments = result.pull_asymmetry_moments[ASYMMETRY_KEYS[0]]
        self.assertEqual(moments.sumw[-2], 1.5)
        self.assertEqual(moments.sumw[-1], 4.0)
        self.assertEqual(moments.zero_sumw, 1.5)
        self.assertEqual(moments.zero_entries, 1)
        np.testing.assert_array_equal(moments.sumw[:-2], np.zeros(60))
        moments.validate()
        statistics = asymmetry_statistics([result], [300.0])[ASYMMETRY_KEYS[0]]
        self.assertEqual(statistics["denominator_pb"], 1.0)
        self.assertEqual(statistics["overflow"]["positive_pb"], 0.5)
        self.assertEqual(statistics["overflow"]["negative_pb"], 0.125)
        self.assertEqual(statistics["zero"]["cross_section_pb"], 0.375)
        self.assertEqual(statistics["zero"]["entries"], 1)

    def test_bin_boundaries_tiny_projections_and_negative_zero(self):
        moments = BeamAsymmetryMoments()
        moments.fill((0.0005, np.nextafter(0.03, 0.0)), 1.0)
        moments.fill((np.nextafter(0.0, 1.0), -0.0), 1.0)
        self.assertEqual(moments.sumw[0], 0.5)
        self.assertEqual(moments.sumw[1], 0.5)
        self.assertEqual(moments.sumw[59], 0.5)
        self.assertEqual(moments.sumw[60], 0.0)
        self.assertEqual(moments.zero_entries, 1)
        self.assertEqual(moments.sumw[-1], 2.0)

    def test_full_second_moments_equal_direct_weighted_event_outer_products(self):
        moments = BeamAsymmetryMoments(np.array([0.0, 0.01, 0.03]))
        moments.fill((0.005, -0.02), 2.0)
        moments.fill((0.005, 0.0), 3.0)
        vectors = [np.array([0.5, -0.5, 0.0, 1.0]),
                   np.array([0.5, 0.0, 0.0, 1.0])]
        weights = [2.0, 3.0]
        np.testing.assert_allclose(moments.sumw, sum(w * v for w, v in zip(weights, vectors)))
        np.testing.assert_allclose(moments.event_second_sumw,
                                   sum(w * np.outer(v, v) for w, v in zip(weights, vectors)))
        np.testing.assert_allclose(moments.mc_second_sumw2,
                                   sum(w * w * np.outer(v, v) for w, v in zip(weights, vectors)))
        self.assertLess(moments.mc_second_sumw2[0, 1], 0.0)
        self.assertGreater(moments.mc_second_sumw2[0, -1], 0.0)

    def test_signed_generator_weights_and_zero_weight_events_are_preserved(self):
        moments = BeamAsymmetryMoments()
        moments.fill((0.00025, 0.00025), 2.0)
        moments.fill((-0.00025, 0.0), -0.5)
        moments.fill((0.0, 0.0), 0.0)
        self.assertEqual(moments.sumw[0], 2.25)
        self.assertEqual(moments.sumw[-1], 1.5)
        self.assertEqual(moments.negative_sumw[0], -0.25)
        self.assertEqual(moments.zero_sumw, -0.25)
        self.assertEqual(moments.zero_entries, 3)
        self.assertEqual(moments.event_count, 3)
        moments.validate()

    def test_shard_merge_and_npz_checkpoint_round_trip(self):
        first = BeamAsymmetryMoments()
        second = BeamAsymmetryMoments()
        first.fill((0.005, -0.015), 2.0)
        second.fill((0.0, 0.04), 3.0)
        combined = BeamAsymmetryMoments()
        combined.fill((0.005, -0.015), 2.0)
        combined.fill((0.0, 0.04), 3.0)
        first.add(second)
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "checkpoint.npz"
            np.savez(archive, **first.to_arrays("sample__pull_asymmetry_beam"))
            with np.load(archive, allow_pickle=False) as arrays:
                restored = BeamAsymmetryMoments.from_arrays(arrays, "sample__pull_asymmetry_beam")
        for attribute in ("sumw", "event_second_sumw", "mc_second_sumw2",
                          "positive_sumw", "negative_sumw"):
            np.testing.assert_array_equal(getattr(restored, attribute), getattr(combined, attribute))
        self.assertEqual(restored.zero_sumw, combined.zero_sumw)
        self.assertEqual(restored.event_count, combined.event_count)
        self.assertEqual(restored.zero_entries, combined.zero_entries)
        # Continue after restoration, as a resumed shard does.
        restored.fill((0.0, 0.0), 1.0)
        self.assertEqual(restored.sumw[-1], 6.0)

    def test_invalid_input_and_corrupt_archives_are_rejected(self):
        moments = BeamAsymmetryMoments()
        for values, weight in [((0.0,), 1.0), ((np.nan, 0.0), 1.0), ((0.0, 0.0), np.inf)]:
            with self.assertRaises(ValueError):
                moments.fill(values, weight)
        with self.assertRaises(ValueError):
            moments.add(BeamAsymmetryMoments(np.array([0.0, 1.0])))
        arrays = moments.to_arrays("test")
        arrays["test__sumw"][0] = 1.0
        with self.assertRaisesRegex(ValueError, "bookkeeping"):
            BeamAsymmetryMoments.from_arrays(arrays, "test")


class BeamAsymmetryStatisticsTests(unittest.TestCase):
    def test_correlated_jets_and_normalization_covariance(self):
        edges = np.array([0.0, 0.01, 0.03])
        result = sample("a", [((0.005, 0.005), 1.0), ((0.02, 0.02), 1.0)],
                        sigma=2.0, edges=edges)
        statistics = asymmetry_statistics([result], [300.0, 3000.0])[ASYMMETRY_KEYS[0]]
        factors = np.array([0.5, 1.0])
        expected_covariance = np.outer(factors, factors) * np.array([[0.125, -0.125], [-0.125, 0.125]])
        np.testing.assert_allclose(statistics["A"], [0.25, 0.5])
        np.testing.assert_allclose(statistics["mc_covariance"], expected_covariance)
        np.testing.assert_allclose(statistics["data_covariances"]["300.0"],
                                   expected_covariance / 300000.0)
        np.testing.assert_allclose(np.asarray(statistics["data_covariances"]["300.0"]),
                                   10.0 * np.asarray(statistics["data_covariances"]["3000.0"]))
        self.assertLess(statistics["mc_covariance"][0][1], 0.0)

    def test_unequal_weight_jacobian_matches_direct_calculation(self):
        edges = np.array([0.0, 0.01, 0.03])
        result = sample("a", [((0.005, -0.02), 2.0), ((0.005, 0.0), 3.0)],
                        sigma=7.0, generated_sumw=8.0, edges=edges)
        moments = result.pull_asymmetry_moments[ASYMMETRY_KEYS[0]]
        factor = 7.0 / 8.0
        first = moments.sumw * factor
        jacobian = np.array([[0.5 / first[-1], 0.0, 0.0, -0.5 * first[0] / first[-1] ** 2],
                             [0.0, 1.0 / first[-1], 0.0, -first[1] / first[-1] ** 2]])
        statistics = asymmetry_statistics([result], [300.0])[ASYMMETRY_KEYS[0]]
        np.testing.assert_allclose(statistics["mc_covariance"],
                                   jacobian @ (factor ** 2 * moments.mc_second_sumw2) @ jacobian.T,
                                   atol=1e-15)
        np.testing.assert_allclose(statistics["data_covariances"]["300.0"],
                                   jacobian @ (factor * moments.event_second_sumw) @ jacobian.T / 300000.0,
                                   atol=1e-15)

    def test_process_stack_closes_with_common_denominator_and_rate_override(self):
        first = sample("first", [((0.00025, 0.00025), 1.0)], sigma=2.0, generated_sumw=2.0)
        second = sample("second", [((-0.00025, 0.0), 3.0)], sigma=3.0)
        payload = asymmetry_statistics([first, second], [300.0])[ASYMMETRY_KEYS[0]]
        self.assertEqual(payload["denominator_pb"], 4.0)
        self.assertEqual(payload["process_contributions"]["first"][0], 0.125)
        self.assertEqual(payload["process_contributions"]["second"][0], -0.1875)
        np.testing.assert_allclose(sum(np.asarray(value) for value in payload["process_contributions"].values()),
                                   payload["A"])
        override = asymmetry_statistics([first, second], [300.0], {"first": 4.0, "second": 3.0})[ASYMMETRY_KEYS[0]]
        self.assertEqual(override["denominator_pb"], 5.0)
        self.assertEqual(override["A"][0], 0.05)
        # The statistics payload is directly exportable without NumPy encoders.
        json.dumps(payload, allow_nan=False)

    def test_tiny_pull_and_all_zero_events_have_no_magnitude_filter(self):
        result = sample("a", [((1e-12, 1e-12), 1.0), ((0.0, 0.0), 1.0)])
        payload = asymmetry_statistics([result], [300.0])[ASYMMETRY_KEYS[0]]
        self.assertEqual(payload["A"][0], 0.25)
        self.assertEqual(payload["denominator_pb"], 1.0)
        self.assertEqual(payload["zero"]["cross_section_pb"], 0.5)
        self.assertFalse(payload["definition"]["pull_magnitude_cut_applied"])

    def test_empty_selection_is_explicit_null_and_missing_archives_fail(self):
        result = sample("empty", [], generated_sumw=100.0)
        payload = asymmetry_statistics([result], [300.0])[ASYMMETRY_KEYS[0]]
        self.assertIsNone(payload["A"])
        self.assertIsNone(payload["mc_covariance"])
        self.assertIsNone(payload["data_covariances"]["300.0"])
        self.assertIsNone(payload["process_contributions"]["empty"])
        self.assertEqual(payload["denominator_pb"], 0.0)
        result.pull_asymmetry_moments = {}
        with self.assertRaisesRegex(ValueError, "reanalyse"):
            asymmetry_statistics([result], [300.0])

    def test_luminosity_independence_and_cross_section_scaling(self):
        result = sample("a", [((0.00025, 0.00025), 2.0), ((-0.00025, 0.0), 1.0)])
        nominal = asymmetry_statistics([result], [300.0, 3000.0])[ASYMMETRY_KEYS[0]]
        scaled = asymmetry_statistics([result], [300.0], {"a": 10.0})[ASYMMETRY_KEYS[0]]
        np.testing.assert_allclose(nominal["A"], scaled["A"])
        np.testing.assert_allclose(nominal["mc_covariance"], scaled["mc_covariance"])
        np.testing.assert_allclose(np.asarray(nominal["data_covariances"]["300.0"]),
                                   10.0 * np.asarray(scaled["data_covariances"]["300.0"]))

    def test_ratio_negative_reference_mask_and_both_error_models(self):
        reference = {"bin_edges": [0.0, 1.0, 2.0, 3.0], "A": [-2.0, 0.0, 4.0],
                     "mc_covariance": np.diag([1.0, 1.0, 4.0]).tolist(),
                     "data_covariances": {"300.0": np.diag([0.01, 0.01, 0.04]).tolist()}}
        variation = {"bin_edges": [0.0, 1.0, 2.0, 3.0], "A": [-4.0, 1.0, 2.0],
                     "mc_covariance": np.diag([4.0, 1.0, 1.0]).tolist(),
                     "data_covariances": {"300.0": np.diag([0.04, 0.01, 0.01]).tolist()}}
        finite_mc = ratio_statistics(variation, reference, "finite_mc")
        self.assertEqual(finite_mc["ratio"], [2.0, None, 0.5])
        self.assertEqual(finite_mc["valid"], [True, False, True])
        self.assertEqual(finite_mc["covariance"][0][0], 2.0)
        self.assertEqual(finite_mc["covariance"][2][2], 0.125)
        self.assertIsNone(finite_mc["covariance"][0][1])
        data = ratio_statistics(variation, reference, "expected_data", 300.0)
        self.assertEqual(data["covariance"][0][0], 0.01)
        self.assertEqual(data["covariance"][2][2], 0.000625)
        json.dumps(data, allow_nan=False)

    def test_tiny_nonzero_ratio_denominators_are_not_significance_masked(self):
        reference = {"bin_edges": [0.0, 1.0, 2.0], "A": [1e-200, -1e-200],
                     "mc_covariance": [[1e-300, 0.0], [0.0, 1e-300]],
                     "data_covariances": {"300.0": [[1e-300, 0.0], [0.0, 1e-300]]}}
        variation = {"bin_edges": [0.0, 1.0, 2.0], "A": [1e-200, -2e-200],
                     "mc_covariance": [[1e-300, 0.0], [0.0, 1e-300]],
                     "data_covariances": {"300.0": [[1e-300, 0.0], [0.0, 1e-300]]}}
        finite_mc = ratio_statistics(variation, reference, "finite_mc")
        np.testing.assert_allclose(finite_mc["ratio"], [1.0, 2.0])
        np.testing.assert_allclose(np.diag(finite_mc["covariance"]), [2e100, 5e100])
        self.assertEqual(finite_mc["valid"], [True, True])
        self.assertEqual(finite_mc["numerical_valid"], [True, True])
        projected = ratio_statistics(variation, reference, "expected_data", 300.0)
        np.testing.assert_allclose(np.diag(projected["covariance"]), [1e100, 1e100])
        json.dumps(finite_mc, allow_nan=False)
        json.dumps(projected, allow_nan=False)

    def test_ratio_arithmetic_overflow_exports_null_separately_from_denominator_mask(self):
        reference = {"bin_edges": [0.0, 1.0, 2.0], "A": [1e-200, 1e-200],
                     "mc_covariance": [[1.0, 0.0], [0.0, 1.0]],
                     "data_covariances": {"300.0": [[0.0, 0.0], [0.0, 0.0]]}}
        variation = {"bin_edges": [0.0, 1.0, 2.0], "A": [1e200, 1e-200],
                     "mc_covariance": [[1.0, 0.0], [0.0, 1.0]],
                     "data_covariances": {"300.0": [[0.0, 0.0], [0.0, 0.0]]}}
        finite_mc = ratio_statistics(variation, reference, "finite_mc")
        self.assertEqual(finite_mc["denominator_valid"], [True, True])
        self.assertEqual(finite_mc["valid"], [True, True])
        self.assertEqual(finite_mc["numerical_valid"], [False, False])
        self.assertEqual(finite_mc["ratio"], [None, 1.0])
        self.assertEqual(finite_mc["errors"], [None, None])
        json.dumps(finite_mc, allow_nan=False)
        projected = ratio_statistics(variation, reference, "expected_data", 300.0)
        self.assertEqual(projected["ratio"], [None, 1.0])
        self.assertEqual(projected["errors"], [None, 0.0])
        json.dumps(projected, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
