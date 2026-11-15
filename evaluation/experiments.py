"""Dev-only analysis using verified baseline rankings and the shared evaluator."""

import csv
import json
import math
import os
import platform
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from .metrics import KS, score_ranking, select_playlists, validate_recommendations
from .run import MODELS, ROOT, evaluate_system, identify, read_json, write_csv

METRICS = tuple(f"{metric}@{k}" for k in KS for metric in ("recall", "ndcg"))
BINS = ("0", "1-5", "6-10", "11-20", "21+")
BLOCK_WEIGHTS = {"tag": 1.0, "genre": 0.1, "decade": 0.2}


def length_bin(count):
    if count < 0:
        raise ValueError("input length cannot be negative")
    if count == 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 10:
        return "6-10"
    if count <= 20:
        return "11-20"
    return "21+"


def averages(rows):
    return {key: math.fsum(r[key] for r in rows) / len(rows) for key in METRICS}


def load_verified_baseline(directory, data_dir, model_path, playlists, counts, catalog_size):
    """Fail closed on stale files, incomplete/duplicate pids or inconsistent cached metrics."""
    summary = read_json(directory / "summary.json")
    if summary["split"] != "dev" or summary["ks"] != list(KS):
        raise ValueError("reused results must be Dev with K=10/20/50/100")
    if summary["catalog_size"] != catalog_size or summary["recommendation_limit"] != 100:
        raise ValueError("reused catalog size or recommendation limit differs")
    if summary["aggregation"] != "playlist macro average":
        raise ValueError("reused aggregation differs")
    if any(summary[key] != value for key, value in counts.items()):
        raise ValueError("reused evaluation counts differ")
    stored = {Path(a["path"]).name: a["sha256"] for a in summary["artifacts"]}
    paths = [data_dir / name for name in ("split.json", "id_mappings.json", "artist_popularity.json",
             "artist_features.npy", "interaction_matrix.npz", "feature_spec.md")] + [model_path]
    artifacts = [identify(path) for path in paths]
    for artifact in artifacts:
        if artifact["sha256"] != stored.get(Path(artifact["path"]).name):
            raise ValueError(f"reused artifact hash mismatch: {artifact['path']}")
    sources = {Path(a["path"]).relative_to(ROOT).as_posix(): a["sha256"]
               for a in summary["source_files"]}
    for name in ("models/content_based/model.py", "models/item_based/model.py", "evaluation/metrics.py"):
        if identify(ROOT / name)["sha256"] != sources.get(name):
            raise ValueError(f"reused source hash mismatch: {name}")
    settings = {s["system"]: s for s in summary["systems"]}
    if set(settings) != set(MODELS):
        raise ValueError("reuse requires exactly the three baseline systems")
    content = settings["content_based"]["settings"]
    if (content.get("weights") != BLOCK_WEIGHTS or content.get("popularity_weight") != 0.3
            or content.get("tie_popularity_epsilon") != 1e-6):
        raise ValueError("reused Content-Based settings differ")
    by_pid = {p["pid"]: p for p in playlists}
    cached = {name: {} for name in MODELS}
    with (directory / "playlists.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            name, pid = row["system"], int(row["pid"])
            if name not in cached or pid not in by_pid or row["split"] != "dev":
                raise ValueError(f"unexpected cached system/pid/split: {name}/{pid}")
            if pid in cached[name]:
                raise ValueError(f"duplicate cached pid: {name}/{pid}")
            p = by_pid[pid]
            ranking = json.loads(row["recommendations"])
            validate_recommendations(ranking, p["input"], catalog_size)
            scores = score_ranking(ranking, p["answer"])
            for key, value in scores.items():
                if not math.isclose(float(row[key]), value, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"cached metric mismatch: {name}/{pid}/{key}")
                row[key] = value
            for key, expected in (("input_artists", len(p["input"])),
                                  ("answer_count", len(set(p["answer"]))),
                                  ("answer_out_of_catalog", p.get("answer_out_of_catalog", 0)),
                                  ("recommendation_count", len(ranking))):
                if int(row[key]) != expected:
                    raise ValueError(f"cached metadata mismatch: {name}/{pid}/{key}")
                row[key] = expected
            row["pid"] = pid
            cached[name][pid] = row
    result = {}
    for name, indexed in cached.items():
        if set(indexed) != set(by_pid):
            raise ValueError(f"incomplete cached pid set: {name}")
        rows = [indexed[p["pid"]] for p in playlists]
        for key, value in averages(rows).items():
            if not math.isclose(value, settings[name][key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"cached summary mismatch: {name}/{key}")
        result[name] = rows
    return summary, result, artifacts


def aggregate_lengths(cached):
    result = []
    for name in MODELS:
        rows = cached[name]
        grouped = {label: [] for label in BINS}
        for row in rows:
            grouped[length_bin(row["input_artists"])].append(row)
        for label, group in grouped.items():
            n = len(group)
            result.append({
                "system": name, "input_bin": label, "playlist_count": n,
                "mean_input_artists": math.fsum(r["input_artists"] for r in group) / n if n else None,
                "mean_answer_artists": math.fsum(r["answer_count"] for r in group) / n if n else None,
                "small_sample_below_30": 0 < n < 30,
                **(averages(group) if n else dict.fromkeys(METRICS)),
            })
        if sum(r["playlist_count"] for r in result if r["system"] == name) != len(rows):
            raise ValueError("length bins do not partition the evaluation set")
        for key, overall in averages(rows).items():
            reconstructed = math.fsum(r[key] * r["playlist_count"] for r in result
                                     if r["system"] == name and r["playlist_count"]) / len(rows)
            if not math.isclose(reconstructed, overall, rel_tol=0, abs_tol=1e-12):
                raise ValueError("length bin weighted reconstruction failed")
    return result


def ablation_comparison(without_rows, with_rows):
    if [r["pid"] for r in without_rows] != [r["pid"] for r in with_rows]:
        raise ValueError("ablation pid sets/order differ")
    without, with_pop = averages(without_rows), averages(with_rows)
    return [{"k": k, "metric": metric, "playlist_count": len(with_rows),
             "popularity_weight_0": without[f"{metric}@{k}"],
             "popularity_weight_0_3": with_pop[f"{metric}@{k}"],
             "absolute_delta": with_pop[f"{metric}@{k}"] - without[f"{metric}@{k}"],
             "relative_change_percent": ((with_pop[f"{metric}@{k}"] / without[f"{metric}@{k}"] - 1) * 100
                                         if without[f"{metric}@{k}"] else None)}
            for k in KS for metric in ("recall", "ndcg")]


def plotting_module():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "playlist-evaluation-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_results(destination, ablation, lengths):
    plt = plotting_module()
    if ablation:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for ax, metric in zip(axes, ("recall", "ndcg")):
            rows = [r for r in ablation if r["metric"] == metric]
            for key, label in (("popularity_weight_0", "Content (weight=0, epsilon remains)"),
                               ("popularity_weight_0_3", "Content + popularity (weight=0.3)")):
                ax.plot(KS, [r[key] for r in rows], marker="o", label=label)
            ax.set(xlabel="K", ylabel=metric.upper() if metric == "ndcg" else "Recall",
                   title=f"{metric.upper()}@K", xticks=KS, ylim=(0, None))
            ax.grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.suptitle("Dev: popularity score ablation (same feature weights)")
        for extension in ("png", "svg"):
            fig.savefig(destination / f"popularity_ablation.{extension}", dpi=180)
        plt.close(fig)
    if lengths:
        labels = [r["input_bin"] for r in lengths if r["system"] == "popularity" and r["playlist_count"]]
        counts = [r["playlist_count"] for r in lengths if r["system"] == "popularity" and r["playlist_count"]]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for ax, metric in zip(axes, ("recall@100", "ndcg@100")):
            for index, name in enumerate(MODELS):
                rows = {r["input_bin"]: r for r in lengths if r["system"] == name}
                ax.bar(np.arange(len(labels)) + (index - 1) * 0.25,
                       [rows[label][metric] for label in labels], width=0.25,
                       label={"popularity": "Popularity", "content_based": "Content + popularity",
                              "item_based": "Item-based"}[name])
            ax.set(xticks=range(len(labels)), xticklabels=[f"{label}\n(n={n:,})" for label, n in zip(labels, counts)],
                   xlabel="Number of input artists", ylabel=metric, title=metric, ylim=(0, None))
            ax.grid(axis="y", alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.suptitle("Dev by input length (groups also differ in answer count and difficulty)")
        for extension in ("png", "svg"):
            fig.savefig(destination / f"input_length.{extension}", dpi=180)
        plt.close(fig)


def run_experiments(args):
    if args.split != "dev":
        raise ValueError("additional experiments are Dev-only; Test evaluation is prohibited")
    if not args.reuse_results:
        raise ValueError("--experiments requires --reuse-results pointing to a baseline run")
    if set(args.models) != set(MODELS):
        raise ValueError("experiments require the three baseline systems; omit --models")
    started, stamp = time.perf_counter(), datetime.now(timezone.utc)
    plotting_module()  # Fail before inference if the optional plotting dependency is absent.
    data_dir, directory = args.data_dir.resolve(), args.reuse_results.resolve()
    playlists, counts = select_playlists(read_json(data_dir / "split.json")["dev"])
    size = len(read_json(data_dir / "id_mappings.json")["artist_ids"])
    baseline, cached, artifacts = load_verified_baseline(
        directory, data_dir, args.model_path.resolve(), playlists, counts, size)
    ablation, lengths, rows, ablation_summary = [], [], [], None
    if "popularity-ablation" in args.experiments:
        from models.content_based.model import DEFAULT_WEIGHTS, DEFAULT_POPULARITY_WEIGHT, load_recommender
        if DEFAULT_WEIGHTS != BLOCK_WEIGHTS or DEFAULT_POPULARITY_WEIGHT != 0.3:
            raise ValueError("current default Content-Based settings changed")
        recommend = load_recommender(data_dir, weights=dict(BLOCK_WEIGHTS), popularity_weight=0.0)
        if recommend.__self__.n_artists != size:
            raise ValueError("ablation feature rows do not match catalog")
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("default")
            rows, ablation_summary = evaluate_system("content_based_popularity_0", recommend, playlists, size)
        ablation_summary.update(
            settings={"weights": dict(BLOCK_WEIGHTS), "popularity_weight": 0.0, "tie_popularity_epsilon": 1e-6},
            runtime_warnings=[{"category": w.category.__name__, "message": str(w.message),
                               "filename": w.filename, "line": w.lineno} for w in observed])
        for w in ablation_summary["runtime_warnings"]:
            print(f"WARNING ablation: {w['message']}", file=sys.stderr)
        ablation = ablation_comparison(rows, cached["content_based"])
    if "input-length" in args.experiments:
        lengths = aggregate_lengths(cached)
    destination = args.output_dir.resolve() / stamp.strftime("experiments_dev_%Y%m%dT%H%M%S_%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    if ablation:
        write_csv(destination / "popularity_ablation.csv", ablation)
        write_csv(destination / "ablation_playlists.csv", rows)
    if lengths:
        write_csv(destination / "input_length.csv", lengths)
    plot_results(destination, ablation, lengths)
    summary = {
        "schema_version": 1, "split": "dev", "started_at_utc": stamp.isoformat(),
        "experiments": args.experiments, **counts, "ks": list(KS),
        "aggregation": "playlist macro average", "artifacts": artifacts,
        "reused_results": [identify(directory / name) for name in ("summary.json", "playlists.csv")],
        "source_files": [identify(ROOT / name) for name in (
            "evaluation/run.py", "evaluation/metrics.py", "evaluation/experiments.py",
            "models/content_based/model.py", "models/item_based/model.py")],
        "baseline_metrics": {name: averages(cached[name]) for name in MODELS},
        "baseline_settings": {s["system"]: s["settings"] for s in baseline["systems"]},
        "baseline_metrics_match_cached_summary": True,
        "new_inference_systems": ["content_based_popularity_0"] if ablation else [],
        "ablation_system": ablation_summary, "popularity_ablation": ablation, "input_length": lengths,
        "input_length_definition": "len(playlist['input'])", "bins": list(BINS),
        "small_sample_threshold": 30,
        "interpretation": [
            "Weight=0 retains 1e-6 popularity correction, and empty inputs still use popularity fallback.",
            "Ablation isolates the 0.3 additive popularity term at fixed feature weights, not feature effectiveness.",
            "Input-length groups differ in answer count and difficulty; differences are not causal effects of length.",
            "SHA-256 identifies bytes, not original checkpoint/mapping provenance."],
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "matplotlib": __import__("matplotlib").__version__,
                        "thread_environment": {key: os.environ.get(key) for key in
                            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
        "total_seconds": time.perf_counter() - started,
    }
    with (destination / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Experiment results: {destination}", flush=True)
    return destination
