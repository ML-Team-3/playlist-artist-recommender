"""
콘텐츠 기반 필터링(Content-Based Filtering) - 플레이리스트 맞춤 아티스트 추천
ML Term Project | Team 03 | 담당: 김택현

아이디어
    "플레이리스트에 이미 들어 있는 아티스트들과 특징(태그·장르·연대)이 비슷한 아티스트를 추천한다."
    강의자료 Content-Based Filtering(114~124쪽)의 구조를 따름
        아이템 = 아티스트, 특징 = artist_features.npy, 사용자 = 플레이리스트

사용하는 전처리 결과 (data_processed 폴더, feature_spec.md 참고)
    artist_features.npy    : 12,778 × 166. 행 번호 = 아티스트 열 번호
                             0–127 태그(TF-IDF → SVD), 128–157 장르 30, 158–165 연대 8
    artist_popularity.json : 열 번호 순서의 Train 등장 플레이리스트 수 (리스트)

처리 순서
    1) 블록(태그/장르/연대)마다 따로 길이를 1로 정규화
       → 블록마다 값의 크기가 달라서, 그대로 이어 붙이면 큰 블록이 유사도를 독차지함
    2) 블록별 가중치를 곱해 이어 붙인 뒤 다시 길이 1로 정규화 (가중치는 Dev로 조정)
    3) 플레이리스트 벡터 = 입력 아티스트 벡터를 song-share로 가중합
    4) 코사인 유사도 높은 순 Top-N (점수가 같으면 Train 인기순)

팀 공통 입출력 규칙
    입력 : split.json의 "input" 그대로  {아티스트 열 번호(str): song-share}
    출력 : 추천 아티스트 열 번호(int) 리스트, 점수 높은 순, 기본 Top 100
           입력에 있는 아티스트 제외, 중복 없음
    학습 : Train만 사용 / 가중치 조정은 Dev / Test는 최종 평가에만 사용

사용하는 라이브러리: numpy, json, pathlib
"""

import json
from pathlib import Path

import numpy as np

# artist_features.npy의 열 구성 (feature_spec.md)
BLOCKS = {
    "tag": slice(0, 128),
    "genre": slice(128, 158),
    "decade": slice(158, 166),
}

# Dev(10,186개 채점)에서 고른 기본값 (tune.py로 재현 가능)
#   태그만                        : Recall@100 0.4328, NDCG@100 0.2203
#   태그 1 / 장르 0.1 / 연대 0.2   : Recall@100 0.4370, NDCG@100 0.2225  (순수 콘텐츠 기반 최고)
#   위 + 인기도 0.3                : Recall@100 0.4823, NDCG@100 0.2543  (기본값)
#   참고: 인기순 베이스라인        : Recall@100 0.1635, NDCG@100 0.0791
DEFAULT_WEIGHTS = {"tag": 1.0, "genre": 0.1, "decade": 0.2}
DEFAULT_POPULARITY_WEIGHT = 0.3


def _normalize_rows(M):
    """행마다 벡터 길이를 1로 맞춤 (길이가 0인 행은 그대로 0)"""
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(norm == 0, 1, norm)


def build_feature_matrix(features, weights):
    """
    블록별 정규화 → 가중치 곱 → 이어 붙임 → 전체 정규화
        features : 아티스트 × 166 원본 특징
        weights  : {"tag": w1, "genre": w2, "decade": w3}  (0이면 그 블록은 사용 안 함)
    """
    parts = [w * _normalize_rows(features[:, BLOCKS[name]])
             for name, w in weights.items() if w > 0]
    if not parts:
        raise ValueError("가중치가 모두 0입니다. 최소 한 블록은 0보다 커야 합니다")
    return _normalize_rows(np.hstack(parts))


class ContentBasedRecommender:
    def __init__(self, features, popularity, weights=None, popularity_weight=DEFAULT_POPULARITY_WEIGHT):
        """
        features          : artist_features.npy (아티스트 × 166)
        popularity        : artist_popularity.json (길이 = 아티스트 수, Train 기준)
        weights           : 블록별 가중치. None이면 DEFAULT_WEIGHTS
        popularity_weight : 유사도 점수에 인기도를 얼마나 더할지 (0이면 동점 정리에만 사용)
                            인기도는 log(1+등장 수)를 0~1로 바꾼 값 → 너무 유명한 가수만 몰리지 않게 완만하게
        """
        features = np.asarray(features, dtype=np.float32)
        popularity = np.asarray(popularity, dtype=np.float64)
        if features.shape[0] != len(popularity):
            raise ValueError("특징 행 수와 인기도 길이가 다릅니다 (열 번호 체계 확인 필요)")

        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.n_artists = features.shape[0]
        self.features = build_feature_matrix(features, self.weights).astype(np.float32)
        log_pop = np.log1p(popularity)
        self.popularity = log_pop / log_pop.max()                         # 0~1
        self.popularity_weight = popularity_weight
        self.popularity_order = np.argsort(-popularity, kind="stable")   # 인기순 열 번호

    def recommend(self, playlist_input, top_n=100):
        """
        playlist_input : {아티스트 열 번호: song-share}  (split.json의 "input" 그대로)
        반환           : 추천 아티스트 열 번호(int) 리스트, 점수 높은 순
        """
        if top_n < 0:
            raise ValueError("top_n은 0 이상이어야 합니다")
        given = {int(k): float(v) for k, v in playlist_input.items()}
        if any(not 0 <= c < self.n_artists for c in given):
            raise ValueError("input에 카탈로그 밖의 열 번호가 있습니다")

        # 입력이 비어 있으면 Train 인기순
        if not given:
            return [int(c) for c in self.popularity_order[:top_n]]

        # 1) 플레이리스트 벡터 = 입력 아티스트 벡터를 song-share로 가중합
        cols = np.fromiter(given.keys(), dtype=np.intp)
        shares = np.fromiter(given.values(), dtype=np.float32)
        profile = shares @ self.features[cols]

        # 2) 코사인 유사도 (아티스트 벡터 길이가 1이므로 내적을 프로필 길이로 나누면 코사인)
        norm = np.linalg.norm(profile)
        scores = (self.features @ profile) / norm if norm > 0 else np.zeros(self.n_artists)

        # 3) 인기도 더하기: popularity_weight만큼 섞고,
        #    0이어도 아주 작은 값(1e-6)을 더해 동점이면 인기 아티스트가 앞에 오게 함
        scores = scores.astype(np.float64) + (self.popularity_weight + 1e-6) * self.popularity

        # 4) 입력 아티스트 제외 후 점수 높은 순 top_n (argsort는 중복 번호를 내지 않음)
        scores[cols] = -np.inf
        limit = min(top_n, self.n_artists - len(cols))
        return [int(c) for c in np.argsort(-scores, kind="stable")[:limit]]


def load_recommender(data_dir, weights=None, popularity_weight=DEFAULT_POPULARITY_WEIGHT):
    """
    평가 코드용: 한 번 불러오고 recommend(playlist_input, top_n=100) 함수를 돌려줌
        data_dir : artist_features.npy, artist_popularity.json이 있는 data_processed 폴더
        weights  : 블록별 가중치. None이면 DEFAULT_WEIGHTS
        popularity_weight : 인기도를 섞는 비중. 기본값은 Dev에서 고른 값
    """
    data_dir = Path(data_dir)
    features = np.load(data_dir / "artist_features.npy")
    with open(data_dir / "artist_popularity.json", encoding="utf-8") as f:
        popularity = json.load(f)
    return ContentBasedRecommender(features, popularity, weights, popularity_weight).recommend
