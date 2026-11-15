"""Binary relevance metrics; callers must validate recommendations first."""

import math
from numbers import Integral

KS = (10, 20, 50, 100)


def validate_recommendations(result, playlist_input, catalog_size, limit=100):
    if not isinstance(result, list):
        raise ValueError("recommendations must be an ordered list")
    if len(result) > limit:
        raise ValueError(f"recommendation length exceeds {limit}")
    seen = set()
    given = {int(c) for c in playlist_input}
    for col in result:
        if isinstance(col, bool) or not isinstance(col, Integral):
            raise ValueError(f"non-integer recommendation: {col!r}")
        if not 0 <= col < catalog_size:
            raise ValueError(f"recommendation outside catalog: {col}")
        if col in seen:
            raise ValueError(f"duplicate recommendation: {col}")
        if col in given:
            raise ValueError(f"input artist recommended: {col}")
        seen.add(col)


def score_ranking(result, answer, ks=KS):
    answer = set(answer)
    if not answer:
        raise ValueError("empty answers must be excluded before scoring")
    scores = {}
    for k in ks:
        if k < 1:
            raise ValueError("K must be positive")
        hits = [rank for rank, col in enumerate(result[:k], 1) if col in answer]
        scores[f"recall@{k}"] = len(hits) / len(answer)
        dcg = sum(1 / math.log2(rank + 1) for rank in hits)
        idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(answer)) + 1))
        scores[f"ndcg@{k}"] = dcg / idcg
    return scores


def select_playlists(playlists):
    selected = [p for p in playlists if p["answer"]]
    if not selected:
        raise ValueError("no playlists with non-empty answers")
    return selected, {
        "total_playlists": len(playlists),
        "excluded_empty_answers": len(playlists) - len(selected),
        "evaluated_playlists": len(selected),
    }
