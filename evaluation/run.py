"""Run shared evaluation: python -m evaluation.run (Dev by default)."""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse

from .metrics import KS, score_ranking, select_playlists, validate_recommendations

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("popularity", "content_based", "item_based")
EXPECTED_COUNTS = {
    "dev": (10257, 71, 10186),
    "test": (10188, 53, 10135),
}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def identify(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def popularity_recommender(popularity):
    # Stable descending order preserves ascending column order on ties.
    order = np.argsort(-np.asarray(popularity, dtype=np.float64), kind="stable").tolist()

    def recommend(playlist_input, top_n=100):
        given = {int(c) for c in playlist_input}
        result = []
        for col in order:
            if len(result) >= top_n:
                break
            if col not in given:
                result.append(col)
        return result

    return recommend


def load_system(name, data_dir, model_path, catalog_size, popularity):
    if name == "popularity":
        return popularity_recommender(popularity), {
            "ranking": "train playlist count descending, column ascending on ties",
        }
    if name == "content_based":
        from models.content_based.model import (
            DEFAULT_POPULARITY_WEIGHT, DEFAULT_WEIGHTS, load_recommender,
        )
        expected = {"tag": 1.0, "genre": 0.1, "decade": 0.2}
        if DEFAULT_WEIGHTS != expected or DEFAULT_POPULARITY_WEIGHT != 0.3:
            raise ValueError("Content-Based defaults differ from the agreed evaluation settings")
        recommend = load_recommender(data_dir)
        if recommend.__self__.n_artists != catalog_size:
            raise ValueError("Content-Based feature rows do not match catalog size")
        return recommend, {
            "weights": dict(DEFAULT_WEIGHTS), "popularity_weight": DEFAULT_POPULARITY_WEIGHT,
            "tie_popularity_epsilon": 1e-6,
        }
    if name == "item_based":
        from models.item_based.model import load_recommender
        graph = sparse.load_npz(model_path)
        if graph.shape != (catalog_size, catalog_size):
            raise ValueError(f"checkpoint shape {graph.shape} does not match catalog")
        settings = {
            "checkpoint_shape": list(graph.shape),
            "max_stored_neighbors_per_row": int(np.diff(graph.indptr).max()),
            "training_neighbors_documented": 200,
            "training_neighbors_source": "README; not checkpoint provenance metadata",
            "popularity_fallback": False,
            "mapping_provenance_verified": False,
        }
        del graph
        return load_recommender(model_path), settings
    raise ValueError(f"unknown model: {name}")


def evaluate_system(name, recommend, playlists, catalog_size):
    started = time.perf_counter()
    rows = []
    for index, playlist in enumerate(playlists, 1):
        try:
            result = recommend(playlist["input"], top_n=100)
            validate_recommendations(result, playlist["input"], catalog_size)
            scores = score_ranking(result, playlist["answer"])
        except Exception as exc:
            raise ValueError(f"{name}, pid={playlist['pid']}: {exc}") from exc
        rows.append({
            "system": name, "pid": playlist["pid"],
            "input_artists": len(playlist["input"]),
            "answer_count": len(set(playlist["answer"])),
            "answer_out_of_catalog": playlist.get("answer_out_of_catalog", 0),
            "recommendation_count": len(result), **scores,
            "recommendations": json.dumps(result, separators=(",", ":")),
        })
        if index % 1000 == 0:
            print(f"{name}: {index}/{len(playlists)}", flush=True)
    metrics = {key: sum(row[key] for row in rows) / len(rows) for key in scores}
    return rows, {
        "system": name, **metrics,
        "average_recommendation_count": sum(r["recommendation_count"] for r in rows) / len(rows),
        "under_100_ratio": sum(r["recommendation_count"] < 100 for r in rows) / len(rows),
        "evaluation_seconds": time.perf_counter() - started,
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--split", choices=("dev", "test"), default="dev")
    result.add_argument("--data-dir", type=Path, default=ROOT / "data_processed")
    result.add_argument("--model-path", type=Path, default=ROOT / "results/item_based/neighbors.npz")
    result.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    result.add_argument("--output-dir", type=Path, default=ROOT / "results/evaluation")
    result.add_argument("--experiments", nargs="+", choices=("popularity-ablation", "input-length"),
                        help="Explicit Dev-only experiments; requires --reuse-results")
    result.add_argument("--reuse-results", type=Path,
                        help="Existing baseline run directory containing summary.json and playlists.csv")
    return result


def run(args):
    if getattr(args, "experiments", None):
        from .experiments import run_experiments
        return run_experiments(args)
    if getattr(args, "reuse_results", None):
        raise ValueError("--reuse-results requires --experiments")
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    data_dir = args.data_dir.resolve()
    model_path = args.model_path.resolve()
    systems = list(dict.fromkeys(args.models))
    playlists, counts = select_playlists(read_json(data_dir / "split.json")[args.split])
    expected = EXPECTED_COUNTS[args.split]
    matches = tuple(counts.values()) == expected
    print(f"{args.split}: {counts}; matches documented counts: {matches}", flush=True)
    if not matches:
        print("WARNING: counts differ from current feature_spec.md; see summary", file=sys.stderr)
    catalog_size = len(read_json(data_dir / "id_mappings.json")["artist_ids"])
    popularity = read_json(data_dir / "artist_popularity.json")
    if len(popularity) != catalog_size:
        raise ValueError("popularity length does not match catalog size")

    files = [data_dir / name for name in (
        "split.json", "id_mappings.json", "artist_popularity.json",
        "artist_features.npy", "interaction_matrix.npz", "feature_spec.md",
    )]
    if "item_based" in systems:
        files.append(model_path)
    artifacts = [identify(path) for path in files]
    source_paths = [Path(__file__), ROOT / "evaluation/metrics.py",
                    ROOT / "models/content_based/model.py", ROOT / "models/item_based/model.py"]
    source_files = [identify(path) for path in source_paths]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    all_rows, summaries = [], []
    for name in systems:
        load_started = time.perf_counter()
        recommend, settings = load_system(name, data_dir, model_path, catalog_size, popularity)
        load_seconds = time.perf_counter() - load_started
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("default")
            rows, summary = evaluate_system(name, recommend, playlists, catalog_size)
        summary["runtime_warnings"] = [
            {"category": warning.category.__name__, "message": str(warning.message),
             "filename": warning.filename, "line": warning.lineno}
            for warning in observed
        ]
        for warning in summary["runtime_warnings"]:
            print(f"WARNING {name}: {warning['message']}", file=sys.stderr)
        for row in rows:
            row["split"] = args.split
        summary.update(split=args.split, **counts, settings=settings, load_seconds=load_seconds)
        all_rows.extend(rows)
        summaries.append(summary)
        print(f"{name}: Recall@100={summary['recall@100']:.8f}, NDCG@100={summary['ndcg@100']:.8f}", flush=True)

    # A new directory per run prevents accidental overwriting/mixing partial results.
    destination = args.output_dir.resolve() / started_at.strftime(f"{args.split}_%Y%m%dT%H%M%S_%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / "playlists.csv", all_rows)
    comparison = [{k: v for k, v in row.items() if k not in ("settings", "runtime_warnings")}
                  for row in summaries]
    write_csv(destination / "comparison.csv", comparison)
    summary = {
        "schema_version": 1, "started_at_utc": started_at.isoformat(),
        "split": args.split, **counts, "documented_counts_match": matches,
        "documented_counts": dict(zip(counts, expected)), "catalog_size": catalog_size,
        "ks": list(KS), "recommendation_limit": 100, "aggregation": "playlist macro average",
        "ground_truth": "binary catalog answer; answer_out_of_catalog excluded from denominator",
        "short_rankings": "missing ranks count as no hits; no fallback",
        "artifacts": artifacts, "source_files": source_files, "git_commit": commit,
        "provenance_note": "SHA-256 identifies file bytes, not original generation history or checkpoint mapping provenance.",
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "thread_environment": {key: os.environ.get(key) for key in
                                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
        "systems": summaries, "total_seconds": time.perf_counter() - started,
    }
    with (destination / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Results: {destination}", flush=True)
    return destination


if __name__ == "__main__":
    run(parser().parse_args())
