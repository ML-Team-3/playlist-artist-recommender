"""Hand-computed metric checks and small real-loader integration tests."""

import argparse
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse

from evaluation.metrics import score_ranking, select_playlists, validate_recommendations
from evaluation.run import evaluate_system, load_system, parser, popularity_recommender, run
from evaluation.experiments import (
    BINS, METRICS, ablation_comparison, aggregate_lengths, length_bin, load_verified_baseline,
)


class MetricTests(unittest.TestCase):
    def test_hand_computed_recall_ndcg(self):
        scores = score_ranking([9, 2, 7, 4], [2, 4, 8], (1, 2, 4))
        self.assertEqual(scores["recall@1"], 0)
        self.assertEqual(scores["recall@2"], 1 / 3)
        self.assertEqual(scores["recall@4"], 2 / 3)
        self.assertAlmostEqual(scores["ndcg@2"], (1 / math.log2(3)) / (1 + 1 / math.log2(3)))
        self.assertAlmostEqual(scores["ndcg@4"], (1 / math.log2(3) + 1 / math.log2(5)) /
                               (1 + 1 / math.log2(3) + 1 / math.log2(4)))

    def test_short_and_empty_recommendations(self):
        scores = score_ranking([2], [2, 3], (10, 100))
        self.assertEqual(scores["recall@100"], 0.5)
        self.assertAlmostEqual(scores["ndcg@100"], 1 / (1 + 1 / math.log2(3)))
        self.assertEqual(scores["ndcg@10"], scores["ndcg@100"])
        self.assertTrue(all(value == 0 for value in score_ranking([], [2]).values()))

    def test_idcg_capped_by_k(self):
        self.assertEqual(score_ranking([2], [2, 3], (1,))["ndcg@1"], 1)

    def test_empty_answer_filter(self):
        selected, counts = select_playlists([{"answer": []}, {"answer": [2]}])
        self.assertEqual(len(selected), 1)
        self.assertEqual(tuple(counts.values()), (2, 1, 1))
        with self.assertRaisesRegex(ValueError, "no playlists"):
            select_playlists([{"answer": []}])
        with self.assertRaisesRegex(ValueError, "empty answers"):
            score_ranking([], [])

    def test_invalid_recommendations(self):
        cases = [([2, 2], "duplicate"), ([1], "input artist"), ([5], "outside catalog"),
                 ([-1], "outside catalog"), ([2.0], "non-integer"), ([True], "non-integer"),
                 ({2}, "ordered list"), ([2, 3, 4], "length exceeds")]
        for result, message in cases:
            with self.subTest(result=result), self.assertRaisesRegex(ValueError, message):
                validate_recommendations(result, {"1": 0.5}, 5, limit=2)
        validate_recommendations([np.int64(2)], {"1": 0.5}, 5)
        validate_recommendations([], {"1": 0.5}, 5)

    def test_popularity_ties_and_exclusion(self):
        recommend = popularity_recommender([1, 9, 9, 3])
        self.assertEqual(recommend({}, 3), [1, 2, 3])
        self.assertEqual(recommend({"1": 0.7}, 100), [2, 3, 0])
        self.assertEqual(recommend({}, 0), [])

    def test_macro_average_and_out_of_catalog(self):
        playlists = [{"pid": 1, "input": {}, "answer": [1], "answer_out_of_catalog": 9},
                     {"pid": 2, "input": {}, "answer": [1, 2, 3]}]
        rows, summary = evaluate_system("fixed", lambda query, top_n: [1], playlists, 5)
        self.assertAlmostEqual(summary["recall@100"], (1 + 1 / 3) / 2)
        self.assertEqual(summary["under_100_ratio"], 1)
        self.assertEqual(len(rows), 2)

    def test_error_context(self):
        with self.assertRaisesRegex(ValueError, "bad, pid=123.*duplicate"):
            evaluate_system("bad", lambda query, top_n: [1, 1],
                            [{"pid": 123, "input": {}, "answer": [1]}], 5)

    def test_cli_defaults(self):
        args = parser().parse_args([])
        self.assertEqual(args.split, "dev")
        self.assertEqual(args.models, ["popularity", "content_based", "item_based"])


class PipelineIntegrationTests(unittest.TestCase):
    def test_real_loaders_and_saved_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = np.zeros((5, 166), dtype=np.float32)
            features[:, 0] = 1
            np.save(root / "artist_features.npy", features)
            (root / "artist_popularity.json").write_text(json.dumps([10, 8, 6, 4, 2]))
            (root / "id_mappings.json").write_text(json.dumps({"artist_ids": list(range(5))}))
            (root / "feature_spec.md").write_text("synthetic fixture")
            sparse.save_npz(root / "interaction_matrix.npz", sparse.eye(5, format="csr"))
            graph = sparse.csr_matrix(([0.9, 0.1], ([0, 1], [1, 2])), shape=(5, 5))
            model_path = root / "neighbors.npz"
            sparse.save_npz(model_path, graph)
            playlists = [{"pid": 1, "input": {"0": 0.2}, "answer": [1], "answer_out_of_catalog": 3},
                         {"pid": 2, "input": {"3": 0.1}, "answer": [2]},
                         {"pid": 3, "input": {}, "answer": []}]
            (root / "split.json").write_text(json.dumps({"dev": playlists}))
            item, _ = load_system("item_based", root, model_path, 5, [])
            self.assertEqual(item({"0": 0.2}), [1])
            self.assertEqual(item({"3": 0.1}), [])  # no implicit fallback
            destination = run(argparse.Namespace(
                split="dev", data_dir=root, model_path=model_path,
                models=["popularity", "content_based", "item_based"], output_dir=root / "out"))
            summary = json.loads((destination / "summary.json").read_text())
            self.assertEqual(summary["excluded_empty_answers"], 1)
            self.assertEqual(summary["evaluated_playlists"], 2)
            self.assertEqual(len(summary["systems"]), 3)
            self.assertEqual(summary["systems"][2]["recall@100"], 0.5)
            self.assertEqual(summary["systems"][2]["average_recommendation_count"], 0.5)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in summary["artifacts"]))
            with (destination / "playlists.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["pid"] for row in rows}, {"1", "2"})
            with (destination / "comparison.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)


class ExperimentTests(unittest.TestCase):
    def test_length_boundaries(self):
        values = (0, 1, 5, 6, 10, 11, 20, 21, 100)
        expected = ("0", "1-5", "1-5", "6-10", "6-10", "11-20", "11-20", "21+", "21+")
        self.assertEqual(tuple(length_bin(value) for value in values), expected)
        with self.assertRaises(ValueError):
            length_bin(-1)

    def test_group_macro_and_empty_bins(self):
        rows = [{"pid": 1, "input_artists": 2, "answer_count": 1,
                 **dict.fromkeys(METRICS, 1.0)},
                {"pid": 2, "input_artists": 5, "answer_count": 3,
                 **dict.fromkeys(METRICS, 1 / 3)},
                {"pid": 3, "input_artists": 21, "answer_count": 5,
                 **dict.fromkeys(METRICS, 0.0)}]
        result = aggregate_lengths({name: rows for name in ("popularity", "content_based", "item_based")})
        group = result[1]
        self.assertEqual(group["playlist_count"], 2)
        self.assertEqual(group["mean_input_artists"], 3.5)
        self.assertEqual(group["mean_answer_artists"], 2)
        self.assertAlmostEqual(group["recall@100"], 2 / 3)
        self.assertTrue(group["small_sample_below_30"])
        self.assertIsNone(result[0]["recall@100"])
        self.assertEqual(sum(r["playlist_count"] for r in result if r["system"] == "popularity"), 3)

    def test_ablation_deltas_and_pid_check(self):
        before = [{"pid": 1, **dict.fromkeys(METRICS, 0.5)}]
        after = [{"pid": 1, **dict.fromkeys(METRICS, 0.75)}]
        result = ablation_comparison(before, after)
        self.assertEqual(len(result), 8)
        self.assertEqual(result[0]["absolute_delta"], 0.25)
        self.assertEqual(result[0]["relative_change_percent"], 50)
        before[0].update(dict.fromkeys(METRICS, 0))
        self.assertIsNone(ablation_comparison(before, after)[0]["relative_change_percent"])
        after[0]["pid"] = 2
        with self.assertRaisesRegex(ValueError, "pid"):
            ablation_comparison(before, after)

    def test_explicit_dev_only_options(self):
        args = parser().parse_args(["--experiments", "input-length", "--split", "test"])
        with self.assertRaisesRegex(ValueError, "Dev-only"):
            run(args)
        args.split = "dev"
        with self.assertRaisesRegex(ValueError, "requires --reuse-results"):
            run(args)
        args = parser().parse_args(["--reuse-results", "unused-results"])
        with self.assertRaisesRegex(ValueError, "requires --experiments"):
            run(args)

    def test_reuse_and_ablation_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = np.zeros((5, 166), dtype=np.float32)
            features[:, 0] = 1
            np.save(root / "artist_features.npy", features)
            for name, obj in (("artist_popularity.json", [10, 8, 6, 4, 2]),
                              ("id_mappings.json", {"artist_ids": list(range(5))})):
                (root / name).write_text(json.dumps(obj))
            (root / "feature_spec.md").write_text("synthetic")
            sparse.save_npz(root / "interaction_matrix.npz", sparse.eye(5, format="csr"))
            model = root / "neighbors.npz"
            sparse.save_npz(model, sparse.eye(5, format="csr"))
            playlists = [{"pid": 1, "input": {"0": 0.2}, "answer": [1]},
                         {"pid": 2, "input": {}, "answer": [2]},
                         {"pid": 3, "input": {}, "answer": []}]
            (root / "split.json").write_text(json.dumps({"dev": playlists}))
            args = parser().parse_args(["--data-dir", str(root), "--model-path", str(model),
                                        "--output-dir", str(root / "out")])
            baseline = run(args)
            baseline_bytes = (baseline / "playlists.csv").read_bytes()
            args.experiments = ["popularity-ablation", "input-length"]
            args.reuse_results = baseline
            from models.content_based.model import load_recommender
            with patch("evaluation.experiments.plotting_module"), patch("evaluation.experiments.plot_results"), \
                 patch("models.content_based.model.load_recommender", wraps=load_recommender) as loader:
                destination = run(args)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(loader.call_args.kwargs, {
                "weights": {"tag": 1.0, "genre": 0.1, "decade": 0.2}, "popularity_weight": 0.0})
            result = json.loads((destination / "summary.json").read_text())
            self.assertEqual(result["new_inference_systems"], ["content_based_popularity_0"])
            self.assertEqual(len(result["input_length"]), 3 * len(BINS))
            self.assertEqual(result["evaluated_playlists"], 2)
            self.assertEqual(result["ablation_system"]["settings"]["tie_popularity_epsilon"], 1e-6)
            self.assertEqual((baseline / "playlists.csv").read_bytes(), baseline_bytes)
            # Length-only analysis never loads or infers a model.
            args.experiments = ["input-length"]
            with patch("evaluation.experiments.plotting_module"), patch("evaluation.experiments.plot_results"), \
                 patch("models.content_based.model.load_recommender", side_effect=AssertionError("inference")):
                length_only = run(args)
            self.assertEqual(json.loads((length_only / "summary.json").read_text())["new_inference_systems"], [])

            selected, counts = select_playlists(playlists)
            def verify():
                return load_verified_baseline(baseline, root, model, selected, counts, 5)
            # Reject corrupted metrics, duplicate/missing pids, and stale artifacts.
            with (baseline / "playlists.csv").open() as stream:
                cached = list(csv.DictReader(stream))
            from evaluation.run import write_csv
            for mode in ("metric", "duplicate", "missing"):
                with self.subTest(mode=mode):
                    changed = [dict(row) for row in cached]
                    if mode == "metric":
                        changed[0]["recall@100"] = "0.123"
                    elif mode == "duplicate":
                        changed.append(dict(changed[0]))
                    else:
                        changed.pop()
                    write_csv(baseline / "playlists.csv", changed)
                    with self.assertRaises(ValueError):
                        verify()
            (baseline / "playlists.csv").write_bytes(baseline_bytes)
            (root / "feature_spec.md").write_text("changed artifact")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                verify()


if __name__ == "__main__":
    unittest.main()
