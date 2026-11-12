"""
콘텐츠 기반 필터링(Content-Based Filtering) - 플레이리스트 맞춤 아티스트 추천
ML Term Project | Team 03

아이디어
    "플레이리스트에 이미 들어 있는 아티스트들과 장르(+태그)가 비슷한 아티스트를 추천한다."
    - 강의자료 Content-Based Filtering(114~124쪽)의 구조를 그대로 따름
        아이템 = 아티스트, 특징(feature) = 장르(+플레이리스트 태그), 사용자 = 플레이리스트

팀 공통 입출력 규칙 (평가 담당과 합의)
    입력 : split.json의 "input" 딕셔너리 그대로  {아티스트 열 번호: song-share}
           (JSON에서 읽으면 키가 문자열 "12"가 되므로 내부에서 int로 바꿔 처리)
    출력 : 추천 아티스트 열 번호(int) 리스트, 점수 높은 순, 기본 Top 100
           입력에 이미 있는 아티스트 제외, 중복 없음
    학습 : Train만 사용 / 파라미터 조정은 Dev / Test는 최종 평가에만 사용

사용하는 라이브러리
    numpy, json (수업에서 사용). TF-IDF와 코사인 유사도는 라이브러리 없이 직접 계산함.
"""

import json
from collections import Counter

import numpy as np


class ContentBasedRecommender:
    """
    파라미터 (Dev 데이터로 조정할 값)
        use_detail_genre : True면 대분류 장르 + 세부 장르, False면 대분류 장르만 사용
        use_tfidf        : True면 TF-IDF 가중치(흔한 장르/태그의 비중을 낮춤), False면 개수 그대로
        tag_weight       : 플레이리스트 태그 특징의 비중 (0이면 태그를 쓰지 않고 장르만 사용)
        top_tags         : 태그를 쓸 때, 자주 쓰인 상위 몇 개 태그만 쓸지
    """

    def __init__(self, use_detail_genre=True, use_tfidf=True, tag_weight=0.0, top_tags=1000):
        self.use_detail_genre = use_detail_genre
        self.use_tfidf = use_tfidf
        self.tag_weight = tag_weight
        self.top_tags = top_tags

    # ------------------------------------------------------------------
    # 학습: 아티스트마다 특징 벡터(장르 점수표)를 만든다
    # ------------------------------------------------------------------
    def fit(self, artist_ids, song_meta, train_playlists=None):
        """
        artist_ids      : 리스트. artist_ids[열 번호] = 원본 아티스트 ID
                          (전처리 담당의 아티스트 열 번호 체계를 그대로 따름)
        song_meta       : song_meta.json을 읽은 리스트 (곡마다 아티스트·장르 정보)
        train_playlists : (선택) Train 플레이리스트 리스트. 각 원소는
                          {"tags": [태그, ...], "artists": [아티스트 열 번호, ...]}
                          - 태그 특징을 쓸 때 필요
                          - 아티스트 인기도(점수가 같을 때 순서 정하기)를 계산할 때 사용
                          ※ 반드시 Train만 넣을 것 (Dev/Test가 섞이면 결과가 부풀려짐)
        """
        self.n_artists = len(artist_ids)
        col_of = {a: i for i, a in enumerate(artist_ids)}   # 원본 아티스트 ID → 열 번호

        # 1) 아티스트 × 장르 개수 세기 ------------------------------------
        genre_counts = [Counter() for _ in range(self.n_artists)]
        for song in song_meta:
            genres = list(song.get("song_gn_gnr_basket", []))
            if self.use_detail_genre:
                genres += song.get("song_gn_dtl_gnr_basket", [])
            for a in song.get("artist_id_basket", []):        # 한 곡에 아티스트가 여러 명일 수 있음
                if a in col_of:
                    genre_counts[col_of[a]].update(genres)

        genre_list = sorted({g for c in genre_counts for g in c})
        g_col = {g: j for j, g in enumerate(genre_list)}
        G = np.zeros((self.n_artists, len(genre_list)))
        for i, c in enumerate(genre_counts):
            for g, n in c.items():
                G[i, g_col[g]] = n
        blocks = [self._weight_and_normalize(G)]

        # 2) (선택) 아티스트 × 태그 개수 세기 -------------------------------
        #    아티스트가 들어 있는 Train 플레이리스트들의 태그를 모음
        if self.tag_weight > 0 and train_playlists:
            tag_freq = Counter(t for p in train_playlists for t in p["tags"])
            tag_list = [t for t, _ in tag_freq.most_common(self.top_tags)]
            t_col = {t: j for j, t in enumerate(tag_list)}
            T = np.zeros((self.n_artists, len(tag_list)))
            for p in train_playlists:
                cols = [t_col[t] for t in p["tags"] if t in t_col]
                for a in set(p["artists"]):
                    T[a, cols] += 1
            blocks.append(self.tag_weight * self._weight_and_normalize(T))

        # 3) 장르 블록과 태그 블록을 이어 붙이고, 다시 길이를 1로 맞춤 ---------
        F = np.hstack(blocks)
        self.features = self._normalize_rows(F)

        # 4) 아티스트 인기도 (점수가 같을 때 순서를 정하는 용도, Train 기준) ----
        pop = np.zeros(self.n_artists)
        if train_playlists:
            for p in train_playlists:
                for a in set(p["artists"]):
                    pop[a] += 1
        self.popularity = pop / pop.max() if pop.max() > 0 else pop
        return self

    # ------------------------------------------------------------------
    # 추천: 팀 공통 형식 recommend(playlist_input, top_n=100)
    # ------------------------------------------------------------------
    def recommend(self, playlist_input, top_n=100):
        """
        playlist_input : {아티스트 열 번호: song-share}  (split.json의 "input" 그대로)
        반환           : 추천 아티스트 열 번호(int) 리스트, 점수 높은 순
        """
        given = {int(k): float(v) for k, v in playlist_input.items()
                 if 0 <= int(k) < self.n_artists}

        # 1) 플레이리스트 벡터 = 들어 있는 아티스트 벡터를 song-share로 가중평균
        profile = np.zeros(self.features.shape[1])
        for a, share in given.items():
            profile += share * self.features[a]
        norm = np.linalg.norm(profile)

        # 2) 모든 아티스트와의 코사인 유사도 (벡터 길이가 1이므로 내적 = 코사인)
        if norm > 0:
            scores = self.features @ (profile / norm)
        else:
            scores = np.zeros(self.n_artists)      # 장르 정보가 없으면 인기순으로만 추천

        # 3) 점수가 같으면 Train 인기도가 높은 아티스트를 앞에 둠 (아주 작은 값만 더함)
        scores = scores + 1e-6 * self.popularity

        # 4) 입력에 이미 있는 아티스트 제외
        for a in given:
            scores[a] = -np.inf

        # 5) 점수 높은 순으로 top_n개 (argsort는 같은 번호를 두 번 내지 않으므로 중복 없음)
        top_n = min(top_n, self.n_artists - len(given))
        top = np.argsort(-scores, kind="stable")[:top_n]
        return [int(i) for i in top]

    # ------------------------------------------------------------------
    # 내부 함수
    # ------------------------------------------------------------------
    def _weight_and_normalize(self, M):
        """개수 행렬 M에 TF-IDF 가중치를 주고 행마다 길이를 1로 맞춤"""
        if self.use_tfidf:
            df = (M > 0).sum(axis=0)                      # 특징마다 그 특징을 가진 아티스트 수
            idf = np.log((1 + M.shape[0]) / (1 + df)) + 1  # 흔한 특징일수록 작은 값
            M = M * idf
        return self._normalize_rows(M)

    @staticmethod
    def _normalize_rows(M):
        norm = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.where(norm == 0, 1, norm)


# ----------------------------------------------------------------------
# 사용 예시 (실제 파일 이름은 전처리 결과에 맞게 바꾸세요)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # 데이터 없이 동작만 확인하는 작은 예제
        artist_ids = [100, 200, 300, 400]          # 열 번호 0~3
        song_meta = [
            {"artist_id_basket": [100], "song_gn_gnr_basket": ["발라드"], "song_gn_dtl_gnr_basket": ["국내발라드"]},
            {"artist_id_basket": [200], "song_gn_gnr_basket": ["발라드"], "song_gn_dtl_gnr_basket": ["국내발라드"]},
            {"artist_id_basket": [300], "song_gn_gnr_basket": ["록"], "song_gn_dtl_gnr_basket": ["모던록"]},
            {"artist_id_basket": [400, 100], "song_gn_gnr_basket": ["발라드"], "song_gn_dtl_gnr_basket": ["OST"]},
        ]
        train = [{"tags": ["새벽"], "artists": [0, 1]}, {"tags": ["드라이브"], "artists": [2]}]
        model = ContentBasedRecommender().fit(artist_ids, song_meta, train)
        print(model.recommend({"0": 1.0}, top_n=100))   # 발라드 가수(열 1, 3)가 록(열 2)보다 앞에 와야 함
    else:
        # python content_based.py song_meta.json artist_ids.json split.json
        with open(sys.argv[1], encoding="utf-8") as f:
            song_meta = json.load(f)
        with open(sys.argv[2], encoding="utf-8") as f:
            artist_ids = json.load(f)
        with open(sys.argv[3], encoding="utf-8") as f:
            split = json.load(f)
        model = ContentBasedRecommender().fit(artist_ids, song_meta)
        first = split["dev"][0] if isinstance(split, dict) and "dev" in split else split[0]
        print(model.recommend(first["input"], top_n=10))
