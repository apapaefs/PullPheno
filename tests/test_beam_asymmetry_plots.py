"""ROOT-independent plotting, ratio propagation and signed-stack closure tests."""
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from beam_asymmetry import ASYMMETRY_KEYS, BeamAsymmetryMoments
import beam_asymmetry_plots as plots


def sample(name, values, *, channel="higgs", color="#275DAD", order=0, sigma=1.0):
    moments = {key: BeamAsymmetryMoments(np.array([0., .001, .002, .03])) for key in ASYMMETRY_KEYS}
    for projections in values:
        for key, accumulator in moments.items():
            accumulator.fill(projections, 1.)
    return SimpleNamespace(
        spec=SimpleNamespace(name=name, channel=channel, label=name, color=color,
                             stack_order=order, cross_section_pb=sigma),
        strategy="cutbased", generated_sumw=float(len(values)), pull_asymmetry_moments=moments,
    )


def source(identifier, results, color="#275DAD"):
    return SimpleNamespace(
        scenario=SimpleNamespace(identifier=identifier, label=identifier, color=color),
        results=results, config=SimpleNamespace(particle_pt_min_gev=.1), metadata={},
    )


class BeamAsymmetryPlotsTests(unittest.TestCase):
    def test_signed_stack_preserves_each_sign_and_algebraic_total(self):
        stack = plots.signed_stack({"signal": [1., -2., 3., -4.], "background": [-2., -1., 2., 1.]})
        np.testing.assert_allclose(stack["bottoms"]["signal"], [0., 0., 0., 0.])
        np.testing.assert_allclose(stack["bottoms"]["background"], [0., -2., 3., 0.])
        np.testing.assert_allclose(stack["positive"], [1., 0., 5., 1.])
        np.testing.assert_allclose(stack["negative"], [-2., -3., 0., -4.])
        np.testing.assert_allclose(stack["total"], [-1., -3., 5., -3.])

    def test_ratio_allows_negative_and_masks_zero_and_nonfinite_reference(self):
        ref = {"A": [-2., 0., np.inf, 4.], "centers": [1., 2., 3., 4.], "bin_edges": [0., 1., 2., 3., 4.],
               "mc_covariance": np.diag([.25, 1., 1., 4.]),
               "data_covariances": {"300.0": np.eye(4)}}
        variation = {"A": [3., 5., 7., 8.], "centers": [1., 2., 3., 4.], "bin_edges": [0., 1., 2., 3., 4.],
                     "mc_covariance": np.array([[1., 0., 0., .2], [0., 1., 0., 0.],
                                                 [0., 0., 1., 0.], [.2, 0., 0., 9.]]),
                     "data_covariances": {"300.0": np.eye(4)}}
        ratio = plots.ratio_statistics(variation, ref, [300.])
        self.assertEqual(ratio["valid"], [True, False, False, True])
        self.assertEqual(ratio["central"], [-1.5, None, None, 2.])
        mc = np.asarray(ratio["mc_covariance"], dtype=float)
        data = np.asarray(ratio["data_covariances"]["300.0"], dtype=float)
        self.assertAlmostEqual(mc[0, 0], .25 + 2.25 * .25 / 4.)
        self.assertAlmostEqual(mc[3, 3], 9./16. + 4. * 4./16.)
        self.assertAlmostEqual(mc[0, 3], -.025)
        self.assertAlmostEqual(data[0, 0], .25)
        self.assertAlmostEqual(data[3, 3], 1./16.)
        self.assertTrue(np.isnan(mc[1]).all())
        self.assertEqual(ratio["reference_mc_relative_error"], [.25, None, None, .5])
        json.dumps(plots._plain(ratio), allow_nan=False)

    def test_common_denominator_stack_and_pairwise_predictions(self):
        nominal = source("herwig-default", [sample("signal", [[.0005, .0015], [.0005, -.0015]]),
                                            sample("background", [[-.0005, -.0015]], sigma=2., order=1)])
        preco = source("herwig-preco025", [sample("signal", [[.0005, .0015]], sigma=5.),
                                          sample("background", [[-.0005, .0015]], sigma=7., order=1)])
        pythia = source("mg5-pythia8-dipolerecoil", [sample("signal", [[-.0005, .0015]]),
                                                   sample("background", [[-.0005, -.0015]], order=1)])
        rates = {"signal": 1., "background": 2.}
        payload = plots.build_comparison_payload([nominal, preco, pythia], ["cutbased"], [300., 3000.], rates)
        pair = plots.build_comparison_payload([nominal, preco], ["cutbased"], [300., 3000.], rates)
        self.assertEqual(pair["predictions"], payload["predictions"][:2])
        self.assertEqual(pair["ratios"], payload["ratios"][:2])
        for prediction in payload["predictions"]:
            for statistics in prediction["observables"].values():
                stack = plots.signed_stack(statistics["process_contributions"])
                np.testing.assert_allclose(stack["total"], statistics["A"])
                self.assertAlmostEqual(statistics["denominator_pb"], 3.)
                np.testing.assert_allclose(statistics["data_covariances"]["3000.0"],
                                           np.asarray(statistics["data_covariances"]["300.0"])/10.)

    def test_near_zero_limits_ignore_outside_view_and_retain_errors(self):
        axis = mock.Mock()
        plots._visible_limits(axis, np.array([.001, .006, .020]), "near-zero",
                              [np.array([-.02, .05, -500.]), np.array([.02, .07, 1000.])])
        lower, upper = axis.set_ylim.call_args.args
        self.assertLess(lower, -.02)
        self.assertGreater(upper, .07)
        self.assertGreater(lower, -.1)
        self.assertLess(upper, .1)

    def test_points_only_omits_process_bars_but_retains_exports(self):
        nominal = source("herwig-default", [sample("signal", [[.0005, -.0015]])])
        variation = source("herwig-preco025", [sample("signal", [[.0005, .0015]])], "#D05A47")
        bar_counts = []
        def inspect_figure(figure, run_dir, relative, particle_pt_min_gev):
            bar_counts.append(sum(patch.__class__.__name__ == "Rectangle" for patch in figure.axes[0].patches))
            return {extension: relative.with_suffix(f".{extension}").as_posix() for extension in ("png", "pdf")}
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(plots, "_save", side_effect=inspect_figure):
            records = plots.generate_comparison_plots(Path(temp), [nominal, variation], ["cutbased"], [300.],
                                                      include_reference_stacks=False)
            self.assertEqual(bar_counts, [0]*8)
            self.assertTrue(all(record["kind"].startswith("asymmetry-total-ratio") for record in records))
            self.assertTrue((Path(temp)/"summaries/asymmetries.json").is_file())

    def test_sample_plots_and_numerical_exports(self):
        result = sample("signal", [[.0005, -.0015], [.0005, .0015]])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = plots.generate_sample_plots(root, [result], [300., 3000.], scenario_label="Herwig",
                                                  particle_pt_min_gev=.1)
            self.assertEqual(len(records), 12)
            self.assertEqual({record["view"] for record in records}, {"full", "near-zero"})
            self.assertTrue(all(record["sample"] == "signal" and record["defined"] for record in records))
            mc_records = [record for record in records if record["error_mode"] == "mc"]
            data_records = [record for record in records if record["error_mode"] == "data"]
            self.assertEqual(len(mc_records), 4)
            self.assertTrue(all(record["kind"] == "asymmetry-shape" and record["luminosity_fb"] is None for record in mc_records))
            self.assertEqual(len(data_records), 8)
            self.assertEqual({record["luminosity_fb"] for record in data_records}, {300., 3000.})
            self.assertTrue(all(record["kind"] == "asymmetry-shape-data-stat" for record in data_records))
            self.assertTrue(all(record["stage"] == "vbf" for record in records))
            for record in records:
                self.assertGreater((root/record["png"]).stat().st_size, 1000)
                self.assertGreater((root/record["pdf"]).stat().st_size, 1000)
            payload = json.loads((root/"summaries/asymmetries.json").read_text())
            self.assertEqual(payload["plots"], records)
            for statistics in payload["predictions"][0]["observables"].values():
                np.testing.assert_allclose(statistics["data_covariances"]["3000.0"],
                                           np.asarray(statistics["data_covariances"]["300.0"])/10.)
                np.testing.assert_allclose(statistics["data_errors"]["3000.0"],
                                           np.asarray(statistics["data_errors"]["300.0"])/np.sqrt(10.))
            with np.load(root/"summaries/asymmetries.npz", allow_pickle=False) as archive:
                np.testing.assert_allclose(archive["prediction_0__pull_asymmetry_beam__A"],
                                           payload["predictions"][0]["observables"]["pull_asymmetry_beam"]["A"])
            with self.assertRaises(FileExistsError):
                plots.write_artifacts(root, payload)

    def test_empty_selected_process_has_explicit_undefined_plots(self):
        result = sample("empty", [[.0005, -.0015]])
        result.pull_asymmetry_moments = {key: BeamAsymmetryMoments(np.array([0., .001, .002, .03]))
                                       for key in ASYMMETRY_KEYS}
        annotations = []
        def inspect_figure(figure, run_dir, relative, particle_pt_min_gev):
            annotations.append([text.get_text() for text in figure.axes[0].texts])
            return {extension: relative.with_suffix(f".{extension}").as_posix() for extension in ("png", "pdf")}
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(plots, "_save", side_effect=inspect_figure):
            records = plots.generate_sample_plots(Path(temp), [result], [300., 3000.])
            self.assertEqual(len(records), 12)
            self.assertTrue(all(record["defined"] is False for record in records))
            self.assertTrue(all(any("Asymmetry undefined" in text for text in texts) for texts in annotations))
            payload = json.loads((Path(temp)/"summaries/asymmetries.json").read_text())
            self.assertTrue(all(statistics["A"] is None for statistics in payload["predictions"][0]["observables"].values()))

    def test_comparison_plot_records_include_errors_views_and_masks(self):
        nominal = source("herwig-default", [sample("signal", [[.0005, -.0015], [.0005, .0015]]),
                                            sample("background", [[-.0005, -.0015]], order=1, color="#D18B20")])
        variation = source("herwig-preco025", [sample("signal", [[.0005, .0015]]),
                                              sample("background", [[-.0005, -.0015]], order=1)], "#D05A47")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = plots.generate_comparison_plots(root, [nominal, variation], ["cutbased"], [300.])
            self.assertEqual(len(records), 8)
            self.assertEqual({record["error_mode"] for record in records}, {"mc", "data"})
            self.assertEqual({record["view"] for record in records}, {"full", "near-zero"})
            self.assertTrue(all(record["luminosity_fb"] == 300. for record in records))
            self.assertEqual({record["kind"] for record in records}, {
                "asymmetry-reference-stack-plus-total-ratio",
                "asymmetry-reference-stack-plus-total-ratio-data-stat"})
            payload = json.loads((root/"summaries/asymmetries.json").read_text())
            self.assertEqual(payload["plots"], records)
            with np.load(root/"summaries/asymmetries.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["ratio_0__valid"], payload["ratios"][0]["valid"])
                self.assertEqual(json.loads(str(archive["metadata_json"])), payload)


if __name__ == "__main__":
    unittest.main()
