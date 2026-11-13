"""Show up to 100 recommendations for an example playlist using the saved model."""

import json
from pathlib import Path

from models.item_based import load_recommender


def main():
    repo = Path(__file__).resolve().parent
    data = repo.parent / "전처리 데이터" / "data_processed"
    with (data / "id_mappings.json").open(encoding="utf-8") as stream:
        names = json.load(stream)["artist_names"]
    with (data / "split.json").open(encoding="utf-8") as stream:
        playlist = json.load(stream)["dev"][0]

    recommend = load_recommender(repo / "results" / "item_based" / "neighbors.npz")
    print("Item-based 아티스트 추천")
    for rank, column in enumerate(recommend(playlist["input"]), 1):
        print(f"{rank}. {names[column]}")


if __name__ == "__main__":
    main()
