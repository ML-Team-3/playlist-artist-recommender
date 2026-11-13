"""Fit and save the item-based collaborative filtering model."""

import argparse
from pathlib import Path

from scipy import sparse

from .model import fit_neighbors, neighbor_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=200, metavar="K")
    parser.add_argument(
        "--output", type=Path, default=Path("results/item_based/neighbors.npz")
    )
    args = parser.parse_args()

    interactions = sparse.load_npz(args.data_dir / "interaction_matrix.npz")
    if not 1 <= args.neighbors < interactions.shape[1]:
        parser.error("--neighbors must be positive and smaller than the artist catalog")
    neighbors, similarities = fit_neighbors(interactions, args.neighbors)
    graph = neighbor_graph(neighbors, similarities, args.neighbors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(args.output, graph)
    print(f"Saved {graph.shape[0]:,} artist neighborhoods (K={args.neighbors}) to {args.output}")


if __name__ == "__main__":
    main()
