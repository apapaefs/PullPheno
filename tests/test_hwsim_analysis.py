import json
import math
import tempfile
import unittest
from unittest import mock
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import HwSimPythonAnalysis as analysis
import xgboost_root_varfiles_module as xgbtools


def particle_event(rows):
    energy, px, py, pz, pid = zip(*rows)
    return analysis.EventParticles(
        np.asarray(energy),
        np.asarray(px),
        np.asarray(py),
        np.asarray(pz),
        np.asarray(pid),
    )


def massless(pt, rapidity, phi, pid):
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(rapidity)
    energy = pt * math.cosh(rapidity)
    return energy, px, py, pz, pid


class FakeJet:
    def __init__(self, pt, rapidity, phi, constituents=()):
        self._pt = pt
        self._rapidity = rapidity
        self._phi = phi
        self._constituents = tuple(constituents)

    def perp(self):
        return self._pt

    def rapidity(self):
        return self._rapidity

    def phi(self):
        return self._phi

    def constituents(self):
        return self._constituents


class SelectionTests(unittest.TestCase):
    def test_delta_phi_wraps_at_pi(self):
        self.assertAlmostEqual(analysis.delta_phi(-math.pi + 0.1, math.pi - 0.1), 0.2)
        self.assertAlmostEqual(analysis.delta_phi(math.pi - 0.1, -math.pi + 0.1), -0.2)

    def test_photon_isolation_excludes_self_and_neutrinos(self):
        rows = [
            massless(50.0, 0.0, 0.0, 22),
            massless(4.0, 0.02, 0.02, 211),
            massless(100.0, 0.01, 0.01, 12),
            massless(20.0, 2.0, 2.0, 211),
        ]
        particles = particle_event(rows)
        self.assertAlmostEqual(particles.relative_photon_isolation(0), 0.08, places=12)

    def test_higgs_candidate_uses_two_leading_isolated_photons(self):
        rows = [
            massless(70.0, 0.0, 0.0, 22),
            massless(55.0, 0.0, math.pi, 22),
            massless(35.0, 1.8, 1.0, 22),
            massless(1.0, -2.0, -1.0, 211),
        ]
        decision = analysis.select_higgs_candidate(particle_event(rows))
        self.assertIsNotNone(decision.candidate)
        self.assertEqual(decision.candidate.leading_index, 0)
        self.assertEqual(decision.candidate.subleading_index, 1)
        self.assertIn("diphoton_mass_window", decision.passed_steps)

    def test_z_candidate_selects_pair_closest_to_z_mass(self):
        rows = [
            massless(45.6, 0.0, 0.0, 11),
            massless(45.6, 0.0, math.pi, -11),
            massless(30.0, 0.0, 1.0, 13),
            massless(25.0, 0.0, -1.0, -13),
        ]
        decision = analysis.select_z_candidate(particle_event(rows))
        self.assertIsNotNone(decision.candidate)
        self.assertEqual({decision.candidate.leading_index, decision.candidate.subleading_index}, {0, 1})
        self.assertEqual(decision.candidate.flavour, "ee")

    def test_vbf_selection(self):
        particles = particle_event(
            [
                massless(100.0, 3.0, 0.0, 1),
                massless(80.0, -3.0, math.pi, 1),
                massless(125.0, 0.0, 0.5, 25),
            ]
        )
        decision = analysis.evaluate_vbf_selection(particles.p4(0), particles.p4(1), particles.p4(2))
        self.assertEqual(decision.passed_steps, ("mjj", "delta_yjj", "boson_centrality"))
        self.assertGreater(decision.mjj, 400.0)
        self.assertEqual(decision.zstar, 0.0)


class PullAndWeightTests(unittest.TestCase):
    def test_positive_and_negative_rapidity_point_to_beam_at_zero_angle(self):
        positive_constituent = FakeJet(10.0, 2.1, 0.0)
        positive = analysis.calculate_pull_vector(FakeJet(100.0, 2.0, 0.0, [positive_constituent]))
        negative_constituent = FakeJet(10.0, -2.1, 0.0)
        negative = analysis.calculate_pull_vector(FakeJet(100.0, -2.0, 0.0, [negative_constituent]))
        self.assertGreater(positive.t_beam, 0.0)
        self.assertGreater(negative.t_beam, 0.0)
        self.assertAlmostEqual(positive.signed_angle, 0.0)
        self.assertAlmostEqual(negative.signed_angle, 0.0)

    def test_phi_component_gives_positive_half_pi(self):
        constituent = FakeJet(10.0, 2.0, 0.1)
        pull = analysis.calculate_pull_vector(FakeJet(100.0, 2.0, 0.0, [constituent]))
        self.assertAlmostEqual(pull.signed_angle, math.pi / 2.0)

    def test_signed_angle_folds_around_zero_and_pi(self):
        self.assertEqual(analysis.fold_signed_pull_angle(0.0), 0.0)
        self.assertAlmostEqual(analysis.fold_signed_pull_angle(-0.7), 0.7)
        self.assertAlmostEqual(analysis.fold_signed_pull_angle(0.7), 0.7)
        self.assertAlmostEqual(analysis.fold_signed_pull_angle(-math.pi), math.pi)
        self.assertAlmostEqual(analysis.fold_signed_pull_angle(math.pi), math.pi)

    def test_symmetric_histogram_fold_adds_sumw_and_sumw2(self):
        source = analysis.WeightedHistogram(np.linspace(-math.pi, math.pi, 13))
        source.sumw[:] = np.arange(1.0, 13.0)
        source.sumw2[:] = np.arange(101.0, 113.0)
        source.entries = 12
        folded = analysis.fold_symmetric_histogram(source)
        np.testing.assert_allclose(folded.edges, np.linspace(0.0, math.pi, 7))
        np.testing.assert_allclose(folded.sumw, [6 + 7, 5 + 8, 4 + 9, 3 + 10, 2 + 11, 1 + 12])
        np.testing.assert_allclose(
            folded.sumw2,
            [106 + 107, 105 + 108, 104 + 109, 103 + 110, 102 + 111, 101 + 112],
        )
        self.assertAlmostEqual(folded.integral, source.integral)
        self.assertEqual(folded.entries, source.entries)

    def test_normalization(self):
        self.assertAlmostEqual(analysis.normalization_factor(300.0, 2.0, 100.0), 6000.0)

    def test_compensated_sum_stabilizes_mixed_event_weights(self):
        weights = [29.316138588567334] + [0.067293999999998078] * 100_000
        naive = 0.0
        compensated = 0.0
        correction = 0.0
        for weight in weights:
            naive += weight
            compensated, correction = analysis.compensated_add(
                compensated,
                correction,
                weight,
            )
        expected = math.fsum(weights)
        self.assertNotEqual(naive, expected)
        self.assertEqual(compensated, expected)

    def test_projected_fbeam_statistical_error_uses_two_jet_entries(self):
        expected = math.sqrt(0.5 * 0.5 / 200.0)
        self.assertAlmostEqual(analysis.projected_fbeam_statistical_error(0.5, 100.0), expected)

    def test_weighted_fraction_mc_error_reduces_to_binomial_error(self):
        self.assertAlmostEqual(
            analysis.weighted_fraction_mc_error(50.0, 100.0, 50.0, 50.0),
            math.sqrt(0.5 * 0.5 / 100.0),
        )

    def test_two_half_weight_entries_preserve_event_weight(self):
        spec = analysis.SampleSpec("VBFH", "higgs", ("unused.root",), 1.0, "VBF H", "#000000", 0)
        result = analysis.initialize_result(spec, 1, 2.5)
        pull = analysis.PullVector(0.01, 0.0, 0.01, 0.01, 0.0, False)
        analysis.fill_pull_histograms(result, (pull, pull), 2.5)
        self.assertAlmostEqual(result.histograms["signed_pull_angle"].integral, 2.5)
        self.assertAlmostEqual(result.histograms["folded_pull_angle"].integral, 2.5)
        self.assertAlmostEqual(result.pull_total_sumw, 2.5)

    def test_channel_pull_summary_uses_physical_sample_normalizations(self):
        first_spec = analysis.SampleSpec("first", "higgs", ("unused.root",), 2.0, "First", "#000000", 0)
        second_spec = analysis.SampleSpec("second", "higgs", ("unused.root",), 1.0, "Second", "#111111", 1)
        first = analysis.initialize_result(first_spec, 10, 10.0)
        second = analysis.initialize_result(second_spec, 20, 20.0)
        first.cutflow["boson_centrality"].sumw = 5.0
        second.cutflow["boson_centrality"].sumw = 10.0
        beam_pull = analysis.PullVector(0.01, 0.0, 0.01, 0.01, 0.0, False)
        away_pull = analysis.PullVector(-0.01, 0.0, -0.01, 0.01, math.pi, False)
        for _ in range(5):
            analysis.fill_pull_histograms(first, (beam_pull, beam_pull), 1.0)
        for _ in range(10):
            analysis.fill_pull_histograms(second, (away_pull, away_pull), 1.0)
        summary = analysis.channel_pull_summary("higgs", (first, second), (300.0,))
        self.assertAlmostEqual(summary["selected_cross_section_pb"], 1.5)
        self.assertAlmostEqual(summary["f_beam"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["expected_selected_yields"]["300.0"], 450000.0)
        self.assertAlmostEqual(
            summary["f_beam_statistical_error"]["300.0"],
            math.sqrt((2.0 / 3.0) * (1.0 / 3.0) / 450000.0),
        )

    def test_differential_ri_closure_and_event_level_covariance(self):
        bin_sums = np.asarray([3.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        event_second = np.diag(bin_sums)
        mc_second = np.diag(bin_sums)
        summary = analysis.differential_pull_statistics(
            bin_sums, event_second, mc_second, (300.0, 3000.0)
        )
        self.assertAlmostEqual(sum(summary["R"]), 1.0)
        self.assertAlmostEqual(summary["f_beam"], sum(summary["R"][:3]))
        expected = np.asarray(summary["expected_statistical_covariance"]["300.0"])
        mc = np.asarray(summary["mc_statistical_covariance"])
        np.testing.assert_allclose(expected.sum(axis=1), 0.0, atol=1.0e-15)
        np.testing.assert_allclose(mc.sum(axis=1), 0.0, atol=1.0e-15)
        self.assertGreaterEqual(np.linalg.eigvalsh(expected).min(), -1.0e-15)
        self.assertGreaterEqual(np.linalg.eigvalsh(mc).min(), -1.0e-15)

    def test_event_bin_vector_keeps_two_jet_correlation(self):
        same = analysis.pull_event_bin_vector((0.0, 0.1))
        split = analysis.pull_event_bin_vector((0.0, math.pi))
        np.testing.assert_allclose(same, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(split, [0.5, 0.0, 0.0, 0.0, 0.0, 0.5])
        np.testing.assert_allclose(np.outer(split, split)[0, 5], 0.25)

    def test_histogram_folds_underflow_and_overflow_and_tracks_sumw2(self):
        histogram = analysis.WeightedHistogram(np.asarray([0.0, 1.0, 2.0]))
        histogram.fill(-4.0, 2.0)
        histogram.fill(8.0, -3.0)
        np.testing.assert_allclose(histogram.sumw, [2.0, -3.0])
        np.testing.assert_allclose(histogram.sumw2, [4.0, 9.0])


class RunManagementTests(unittest.TestCase):
    def test_run_id_and_collision_suffix_are_non_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            now = datetime(2026, 8, 11, 12, 30, tzinfo=timezone.utc)
            base = analysis.make_run_id({"answer": 42}, "nominal run", now)
            first = analysis.reserve_run_directory(root, base)
            second = analysis.reserve_run_directory(root, base)
            self.assertNotEqual(first.run_id, second.run_id)
            self.assertTrue(first.incomplete_dir.is_dir())
            self.assertTrue(second.incomplete_dir.is_dir())
            self.assertRegex(first.run_id, r"nominal-run-[0-9a-f]{8}$")
            self.assertEqual(second.run_id, first.run_id + "-01")

    def test_exclusive_artifact_write_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.txt"
            analysis.write_text_exclusive(path, "first")
            with self.assertRaises(FileExistsError):
                analysis.write_text_exclusive(path, "second")
            self.assertEqual(path.read_text(), "first")

    def test_catalog_excludes_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = root / "runs" / "complete-run"
            incomplete = root / "runs" / ".incomplete-other"
            complete.mkdir(parents=True)
            incomplete.mkdir()
            (complete / "index.html").write_text("<!doctype html><title>run</title>")
            (complete / "run.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "run_id": "complete-run",
                        "completed_utc": "2026-08-11T12:00:00+00:00",
                        "partial": False,
                        "samples": [],
                    }
                )
            )
            (incomplete / "run.json").write_text(json.dumps({"status": "complete"}))
            index, runs_json = analysis.update_top_level_catalog(root)
            catalog = json.loads(runs_json.read_text())
            self.assertEqual([item["run_id"] for item in catalog], ["complete-run"])
            self.assertIn("runs/complete-run/index.html", index.read_text())


class XGBoostHelperTests(unittest.TestCase):
    def test_feature_list_excludes_pull_and_colour_information(self):
        self.assertEqual(
            xgbtools.FEATURE_NAMES,
            (
                "mjj",
                "abs_delta_yjj",
                "leading_jet_pt",
                "subleading_jet_pt",
                "boson_pt",
                "zstar",
            ),
        )
        forbidden = ("pull", "constituent", "gap", "colour", "reconnection")
        self.assertFalse(any(word in feature for feature in xgbtools.FEATURE_NAMES for word in forbidden))

    def test_deterministic_split_is_exact_disjoint_and_reproducible(self):
        first = xgbtools.deterministic_split(101, "VBFH", seed=1)
        second = xgbtools.deterministic_split(101, "VBFH", seed=1)
        self.assertEqual((len(first.train), len(first.validation), len(first.test)), (60, 20, 21))
        np.testing.assert_array_equal(first.train, second.train)
        combined = np.concatenate((first.train, first.validation, first.test))
        np.testing.assert_array_equal(np.sort(combined), np.arange(101))

    def test_five_fold_crossfit_is_disjoint_exhaustive_and_leakage_free(self):
        first = xgbtools.deterministic_crossfit_splits(103, "VBFH", folds=5, seed=1)
        second = xgbtools.deterministic_crossfit_splits(103, "VBFH", folds=5, seed=1)
        self.assertEqual(len(first), 5)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.train, right.train)
            np.testing.assert_array_equal(left.validation, right.validation)
            np.testing.assert_array_equal(left.test, right.test)
            self.assertFalse(set(left.train) & set(left.validation))
            self.assertFalse(set(left.train) & set(left.test))
            self.assertFalse(set(left.validation) & set(left.test))
            np.testing.assert_array_equal(
                np.sort(np.concatenate((left.train, left.validation, left.test))),
                np.arange(103),
            )
        test_coverage = np.concatenate([split.test for split in first])
        validation_coverage = np.concatenate([split.validation for split in first])
        training_coverage = np.concatenate([split.train for split in first])
        np.testing.assert_array_equal(np.sort(test_coverage), np.arange(103))
        np.testing.assert_array_equal(np.sort(validation_coverage), np.arange(103))
        np.testing.assert_array_equal(
            np.bincount(training_coverage, minlength=103), np.full(103, 3)
        )

    def test_five_fold_crossfit_rejects_empty_folds(self):
        with self.assertRaises(ValueError):
            xgbtools.deterministic_crossfit_splits(4, "tiny", folds=5, seed=1)

    def test_training_weights_balance_classes_and_preserve_ratios(self):
        physical = np.asarray([1.0, 3.0, 10.0, 20.0])
        labels = np.asarray([1, 1, 0, 0])
        balanced, metadata = xgbtools.balanced_training_weights(physical, labels)
        self.assertAlmostEqual(metadata["signal_sum"], 2.0)
        self.assertAlmostEqual(metadata["background_sum"], 2.0)
        self.assertAlmostEqual(balanced[1] / balanced[0], 3.0)
        self.assertAlmostEqual(balanced[3] / balanced[2], 2.0)

    def test_threshold_maximizes_physical_significance_and_ties_are_tight(self):
        scores = np.asarray([0.9, 0.8, 0.7, 0.6])
        labels = np.asarray([1, 0, 1, 0])
        weights = np.ones(4)
        optimum = xgbtools.optimize_significance_threshold(scores, labels, weights)
        self.assertEqual(optimum.threshold, 0.7)
        self.assertAlmostEqual(optimum.significance, 2.0 / math.sqrt(3.0))

    def test_xgboost_application_does_not_require_vbf_cuts(self):
        spec = analysis.SampleSpec(
            "VBFH", "higgs", ("unused.root",), 1.0, "VBF H", "#000", 0, "signal"
        )
        cut = analysis.initialize_result(spec, 1, 1.0)
        for step in analysis.HIGGS_CUTFLOW[:-3]:
            cut.cutflow[step].fill(1.0)
        keys = analysis.common_observable_keys("higgs")
        values = {key: 1.0 for key in keys}
        values.update({"mjj": 100.0, "abs_delta_yjj": 0.5, "zstar": 2.0})
        pull_values = np.asarray([[[0.01, 0.0, 0.01, 0.0, 0.0], [0.01, 0.0, 0.01, 0.0, 0.0]]])
        cut.common_events = analysis.CommonEventTable(
            observable_keys=keys,
            weights=np.asarray([1.0]),
            observables=np.asarray([[values[key] for key in keys]]),
            pulls=pull_values,
            source_file_indices=np.asarray([0], dtype=np.int32),
            source_entries=np.asarray([0], dtype=np.int64),
        )
        selected = analysis._fill_xgboost_sample_result(
            cut, np.asarray([0]), np.asarray([0.9]), threshold=0.5, correction=1.0
        )
        self.assertEqual(selected.cutflow["xgboost_score"].raw_count, 1)
        self.assertNotIn("mjj", selected.cutflow)
        self.assertNotIn("delta_yjj", selected.cutflow)
        self.assertNotIn("boson_centrality", selected.cutflow)

    def test_per_event_fold_thresholds_are_applied_without_inverse_weights(self):
        spec = analysis.SampleSpec(
            "VBFH", "higgs", ("unused.root",), 1.0, "VBF H", "#000", 0, "signal"
        )
        cut = analysis.initialize_result(spec, 2, 2.0)
        for step in analysis.HIGGS_CUTFLOW[:-3]:
            cut.cutflow[step].fill(1.0)
            cut.cutflow[step].fill(1.0)
        keys = analysis.common_observable_keys("higgs")
        values = {key: 1.0 for key in keys}
        pull_values = np.asarray(
            [
                [[0.01, 0.0, 0.01, 0.0, 0.0], [0.01, 0.0, 0.01, 0.0, 0.0]],
                [[0.01, 0.0, 0.01, 0.0, 0.0], [0.01, 0.0, 0.01, 0.0, 0.0]],
            ]
        )
        cut.common_events = analysis.CommonEventTable(
            observable_keys=keys,
            weights=np.asarray([1.0, 1.0]),
            observables=np.asarray([[values[key] for key in keys]] * 2),
            pulls=pull_values,
            source_file_indices=np.asarray([0, 0], dtype=np.int32),
            source_entries=np.asarray([0, 1], dtype=np.int64),
        )
        selected = analysis._fill_xgboost_sample_result(
            cut,
            np.asarray([0, 1]),
            np.asarray([0.8, 0.8]),
            threshold=np.asarray([0.7, 0.9]),
            correction=1.0,
        )
        self.assertEqual(selected.cutflow["xgboost_application_sample"].raw_count, 2)
        self.assertEqual(selected.cutflow["xgboost_application_sample"].sumw, 2.0)
        self.assertEqual(selected.cutflow["xgboost_score"].raw_count, 1)
        self.assertEqual(selected.cutflow["xgboost_score"].sumw, 1.0)

    def test_crossfit_build_scores_every_nominal_event_once(self):
        class FakeBooster:
            def get_score(self, importance_type="gain"):
                return {f"f{index}": float(index + 1) for index in range(len(xgbtools.FEATURE_NAMES))}

        class FakeClassifier:
            def predict_proba(self, features):
                score = np.clip(np.asarray(features)[:, 0] / 1000.0, 0.0, 1.0)
                return np.column_stack((1.0 - score, score))

            def get_booster(self):
                return FakeBooster()

        def make_result(name, channel, role):
            event_count = 10
            spec = analysis.SampleSpec(
                name=name,
                channel=channel,
                files=("unused.root",),
                cross_section_pb=1.0,
                label=name,
                color="#000000",
                stack_order=0,
                role=role,
            )
            result = analysis.initialize_result(spec, event_count, float(event_count))
            result.processed_entries = event_count
            result.processed_sumw = float(event_count)
            for step in analysis.cutflow_steps(channel, "cutbased")[:-3]:
                for _ in range(event_count):
                    result.cutflow[step].fill(1.0)
            keys = analysis.common_observable_keys(channel)
            rows = []
            for index in range(event_count):
                values = {key: 1.0 for key in keys}
                values.update(
                    {
                        "mjj": (800.0 if role == "signal" else 100.0) + index,
                        "abs_delta_yjj": 3.0 if role == "signal" else 1.0,
                        "leading_jet_pt": 100.0,
                        "subleading_jet_pt": 60.0,
                        "boson_pt": 80.0,
                        "zstar": 0.2,
                    }
                )
                rows.append([values[key] for key in keys])
            pull = np.asarray([0.01, 0.0, 0.01, 0.2, 0.0])
            result.common_events = analysis.CommonEventTable(
                observable_keys=keys,
                weights=np.ones(event_count),
                observables=np.asarray(rows),
                pulls=np.tile(pull, (event_count, 2, 1)),
                source_file_indices=np.zeros(event_count, dtype=np.int32),
                source_entries=np.arange(event_count, dtype=np.int64),
            )
            return result

        inputs = [
            make_result("H-background", "higgs", "background"),
            make_result("H-signal", "higgs", "signal"),
            make_result("Z-background", "z", "background"),
            make_result("Z-signal", "z", "signal"),
        ]
        fake = FakeClassifier()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            xgbtools, "train_classifier", return_value=fake
        ) as train, mock.patch.object(
            xgbtools, "save_classifier", return_value="a" * 64
        ), mock.patch.object(
            xgbtools, "load_classifier", return_value=fake
        ):
            results, metadata, diagnostics = analysis.build_xgboost_results(
                inputs, Path(temporary) / "nominal-output"
            )
            source_run = Path(temporary) / "source-run"
            (source_run / "summaries").mkdir(parents=True)
            (source_run / "summaries" / "xgboost.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            frozen_results, frozen_metadata, _ = analysis.build_xgboost_results(
                inputs,
                Path(temporary) / "variation-output",
                model_run=source_run,
            )
            html_run = Path(temporary) / "html-run"
            html_run.mkdir()
            index = analysis.generate_run_index(
                html_run,
                {
                    "run_id": "synthetic-five-fold",
                    "completed_utc": "2026-08-11T00:00:00+00:00",
                    "partial": False,
                },
                results,
                [analysis.result_summary(result, (300.0,)) for result in results],
                [],
                (300.0,),
                metadata,
            )
            html_text = index.read_text(encoding="utf-8")
        self.assertEqual(train.call_count, 10)
        self.assertEqual(len(results), 4)
        self.assertEqual(analysis.validate_results(results, partial=False)["status"], "passed")
        for result in results:
            self.assertEqual(result.application_scope, "five_fold_out_of_fold_all_events")
            self.assertEqual(result.cutflow["xgboost_application_sample"].raw_count, 10)
            self.assertEqual(result.cutflow["xgboost_application_sample"].sumw, 10.0)
        for channel in ("higgs", "z"):
            channel_metadata = metadata["channels"][channel]
            self.assertEqual(channel_metadata["model_count"], 5)
            self.assertEqual(len(channel_metadata["score_thresholds"]), 5)
            self.assertTrue(channel_metadata["reload_predictions_identical"])
            self.assertEqual(channel_metadata["out_of_fold"]["scope"], "five_fold_out_of_fold_all_events")
            for sample in channel_metadata["samples"].values():
                self.assertEqual(sample["input_count"], 10)
                self.assertEqual(sample["inverse_probability"], 1.0)
                self.assertEqual(sum(sample["fold_input_counts"].values()), 10)
            self.assertIn("out-of-fold", diagnostics[channel]["splits"])
            frozen_channel = frozen_metadata["channels"][channel]
            self.assertEqual(
                frozen_channel["application_scope"],
                "five_fold_routed_independent_events",
            )
            self.assertEqual(frozen_channel["model_count"], 5)
            self.assertIn("application", frozen_channel)
        self.assertIn("rotating five-fold out-of-fold predictions", html_text)
        self.assertIn("fold 5", html_text)
        for result in frozen_results:
            self.assertEqual(
                result.application_scope, "five_fold_routed_independent_events"
            )
            self.assertEqual(result.cutflow["xgboost_application_sample"].raw_count, 10)
            self.assertEqual(result.cutflow["xgboost_application_sample"].sumw, 10.0)
        self.assertEqual(
            analysis.validate_results(frozen_results, partial=False)["status"], "passed"
        )

    def test_model_serialization_preserves_probabilities_when_available(self):
        try:
            import xgboost  # noqa: F401
        except ImportError:
            self.skipTest("xgboost is optional in the local cut-based environment")
        generator = np.random.default_rng(12)
        features = generator.normal(size=(80, len(xgbtools.FEATURE_NAMES)))
        labels = np.r_[np.zeros(40, dtype=np.int8), np.ones(40, dtype=np.int8)]
        weights = np.ones(80)
        classifier = xgbtools.train_classifier(features, labels, weights)
        expected = xgbtools.signal_scores(classifier, features)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            digest = xgbtools.save_classifier(classifier, path)
            restored = xgbtools.load_classifier(path)
            self.assertEqual(len(digest), 64)
            np.testing.assert_array_equal(
                expected, xgbtools.signal_scores(restored, features)
            )


if __name__ == "__main__":
    unittest.main()
