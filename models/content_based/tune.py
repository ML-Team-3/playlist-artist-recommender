"""
Content-based 가중치 조정 (Dev 전용)

블록 가중치(태그/장르/연대)와 인기도 가중치를 바꿔 가며 Dev에서
Recall@100, NDCG@100을 계산한다. 채점 규칙은 feature_spec.md를 따른다.
    - answer가 빈 플레이리스트는 건너뜀
    - Recall@100 = 맞힌 정답 수 / 정답 수
    - NDCG@100   = 맞힌 위치에 따른 점수 / 가장 이상적인 점수 (정답 = 1점)
※ Test는 최종 평가용이므로 여기서 쓰지 않는다.

실행 (저장소 맨 위 폴더에서):
    python -m models.content_based.tune --data-dir "../전처리 데이터/data_processed"
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .model import ContentBasedRecommender

CONFIGS = [  # (블록 가중치, 인기도 가중치)
    ({"tag": 1, "genre": 0, "decade": 0}, 0.0),
    ({"tag": 0, "genre": 1, "decade": 0}, 0.0),
    ({"tag": 0, "genre": 0, "decade": 1}, 0.0),
    ({"tag": 1, "genre": 1, "decade": 1}, 0.0),
    ({"tag": 1, "genre": 0.1, "decade": 0.2}, 0.0),
    ({"tag": 1, "genre": 0.1, "decade": 0.2}, 0.1),
    ({"tag": 1, "genre": 0.1, "decade": 0.2}, 0.3),
    ({"tag": 1, "genre": 0.1, "decade": 0.2}, 0.6),
]


def evaluate(recommend, playlists, k=100):
    """Recall@k, NDCG@k 평균"""
    discount = 1 / np.log2(np.arange(2, k + 2))       # 순위 1, 2, 3... 의 할인값
    recall = ndcg = 0.0
    for p in playlists:
        answer = set(p["answer"])
        hits = np.array([c in answer for c in recommend(p["input"], top_n=k)], dtype=float)
        recall += hits.sum() / len(answer)
        ideal = discount[: min(len(answer), k)].sum()
        ndcg += (hits * discount[: len(hits)]).sum() / ideal
    return recall / len(playlists), ndcg / len(playlists)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    features = np.load(args.data_dir / "artist_features.npy")
    with (args.data_dir / "artist_popularity.json").open(encoding="utf-8") as f:
        popularity = json.load(f)
    with (args.data_dir / "split.json").open(encoding="utf-8") as f:
        dev = [p for p in json.load(f)["dev"] if p["answer"]]
    print(f"Dev 채점 대상 {len(dev):,}개")

    for weights, pop_weight in CONFIGS:
        model = ContentBasedRecommender(features, popularity, weights, pop_weight)
        recall, ndcg = evaluate(model.recommend, dev)
        print(f"{weights} 인기도 {pop_weight}: Recall@100 {recall:.4f}  NDCG@100 {ndcg:.4f}", flush=True)


if __name__ == "__main__":
    main()
