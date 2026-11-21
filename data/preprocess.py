"""
멜론 플레이리스트 → 아티스트 추천용 데이터 전처리 (담당: 장승우)

실행 (저장소 루트에서): python data/preprocess.py
입력: data_raw/ (Kakao Arena에서 받은 train.json, song_meta.json, genre_gn_all.json)
출력: data_processed/ (파일 설명은 실행 후 생성되는 feature_spec.md 참고)

데이터 출처: Kakao Arena Melon Playlist Dataset (https://kakao.github.io/recoteam/arena/melon)
팀 결정 사항 (2026-11-02):
  - 분할: 플레이리스트 단위로 train 80 / dev 10 / test 10 (팀 제안서 방식)
  - dev/test 플레이리스트에서 아티스트 20%(최소 1명)를 정답으로 숨김
  - 채점 대상: 카탈로그 아티스트가 5명 이상인 dev/test 플레이리스트
  - val.json / test.json(대회 파일)은 정답이 없어 사용하지 않음 (시연 입력용)
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
OUT = ROOT / "data_processed"
MIN_SONGS = 5          # 이보다 곡이 적은 플레이리스트는 취향 신호가 약해 제거
MIN_ARTIST_PL = 20     # 학습 플레이리스트 20개 이상에 등장한 아티스트만 카탈로그에 포함
MIN_TAG_COUNT = 5      # 5번 미만 등장한 태그는 노이즈로 보고 제거
HIDE_RATIO = 0.2       # dev/test 플레이리스트에서 정답으로 숨길 아티스트 비율
MIN_EVAL_ARTISTS = 5   # 채점 대상이 되려면 카탈로그 아티스트가 이만큼 있어야 함
TAG_DIM = 128
# 연대 8구간: 발매연도가 [시작, 끝) 범위에 들어가면 해당 구간
ERAS = [(0, 1970), (1970, 1980), (1980, 1990), (1990, 2000),
        (2000, 2005), (2005, 2010), (2010, 2015), (2015, 9999)]


def load(name):
    with open(RAW / name, encoding="utf-8") as f:
        return json.load(f)


# ---------- 1단계: 곡 → 아티스트·장르·연대 조인 ----------
print("1/6 song_meta 불러오는 중 (217MB, 30초 정도 걸림)")
genre_codes = sorted(k for k in load("genre_gn_all.json") if k.endswith("00"))  # 대분류 30종
genre_idx = {g: i for i, g in enumerate(genre_codes)}

song_artists = {}             # 곡 id → 아티스트 id 목록
artist_name = {}
various = set()               # 컴필레이션용 'Various Artists'는 실제 아티스트가 아니므로 제외
artist_genre = defaultdict(lambda: np.zeros(len(genre_codes)))
artist_era = defaultdict(lambda: np.zeros(len(ERAS)))
for s in load("song_meta.json"):
    arts = []
    for a, n in zip(s["artist_id_basket"], s["artist_name_basket"]):
        if n == "Various Artists":
            various.add(a)
            continue
        arts.append(a)
        artist_name[a] = n
    song_artists[s["id"]] = arts
    year = int(s["issue_date"][:4]) if s["issue_date"][:4].isdigit() else 0
    for a in arts:
        for g in s["song_gn_gnr_basket"]:
            if g in genre_idx:
                artist_genre[a][genre_idx[g]] += 1
        if year >= 1900:  # 0000 같은 잘못된 발매일은 건너뜀
            artist_era[a][next(i for i, (lo, hi) in enumerate(ERAS) if lo <= year < hi)] += 1

# ---------- 2단계: 중복·짧은 플레이리스트 제거 ----------
# 분할보다 먼저 해야 같은 곡 목록이 train과 dev/test에 동시에 들어가지 않음
print("2/6 플레이리스트 정리")
raw_train = load("train.json")
seen, playlists, removed = set(), [], Counter()
for p in raw_train:
    key = tuple(sorted(p["songs"]))
    dup = key in seen
    seen.add(key)
    if dup or len(p["songs"]) < MIN_SONGS:
        removed["duplicate" if dup else "short"] += 1
        continue
    # 아티스트별 곡 수 비율(song-share): 그 플레이리스트에서 이 아티스트 곡이 차지하는 비중
    cnt = Counter(a for s in p["songs"] for a in set(song_artists[s]))
    p["share"] = {a: c / len(p["songs"]) for a, c in cnt.items()}
    playlists.append(p)

# ---------- 3단계: 플레이리스트 단위 분할 ----------
print("3/6 train/dev/test 분할")
rng = np.random.default_rng(SEED)
order = rng.permutation(len(playlists))
n_dev = n_test = len(playlists) // 10
parts = {
    "dev": [playlists[i] for i in order[:n_dev]],
    "test": [playlists[i] for i in order[n_dev:n_dev + n_test]],
    "train": [playlists[i] for i in order[n_dev + n_test:]],
}
train = parts["train"]

# ---------- 4단계: 아티스트 카탈로그 (train만 보고 결정 → 평가 누수 방지) ----------
print("4/6 아티스트 카탈로그")
pl_count = Counter(a for p in train for a in p["share"])
catalog = sorted(a for a, c in pl_count.items() if c >= MIN_ARTIST_PL)
col = {a: j for j, a in enumerate(catalog)}

# 학습용 상호작용 행렬: 행 = train 플레이리스트, 열 = 카탈로그 아티스트, 값 = song-share
rows, cols, vals = [], [], []
for i, p in enumerate(train):
    for a, w in p["share"].items():
        if a in col:
            rows.append(i)
            cols.append(col[a])
            vals.append(w)
# scipy.sparse.csr_matrix: 0이 대부분인 행렬을 0이 아닌 값만 저장하는 희소 행렬 형식
X = sp.csr_matrix((vals, (rows, cols)), shape=(len(train), len(catalog)), dtype=np.float32)

# dev/test 시험지: 아티스트 20%를 숨기고 나머지를 입력으로 줌
eval_sets, coverage = {}, {}
for name in ("dev", "test"):
    items, hidden_total, hidden_in_cat = [], 0, 0
    for p in parts[name]:
        arts = sorted(p["share"])
        if sum(a in col for a in arts) < MIN_EVAL_ARTISTS:
            continue
        k = max(1, round(len(arts) * HIDE_RATIO))
        hidden = set(rng.choice(arts, size=k, replace=False).tolist())
        answer = [col[a] for a in hidden if a in col]
        hidden_total += len(hidden)
        hidden_in_cat += len(answer)
        items.append({
            "pid": p["id"],
            "input": {str(col[a]): round(p["share"][a], 4) for a in arts if a not in hidden and a in col},
            "answer": sorted(answer),
            "answer_out_of_catalog": len(hidden) - len(answer),
            "tags": p["tags"],
        })
    eval_sets[name] = items
    coverage[name] = hidden_in_cat / hidden_total

# ---------- 5단계: 아티스트 특성 벡터 ----------
print("5/6 아티스트 특성 (태그 SVD + 장르 + 연대)")
tag_count = Counter(t for p in train for t in p["tags"])
tag_vocab = sorted(t for t, c in tag_count.items() if c >= MIN_TAG_COUNT)
tag_col = {t: j for j, t in enumerate(tag_vocab)}
rows, cols = [], []
for p in train:
    tj = [tag_col[t] for t in p["tags"] if t in tag_col]
    for a in p["share"]:
        if a in col:
            rows += [col[a]] * len(tj)
            cols += tj
# 아티스트 × 태그 등장 횟수 (같은 칸이 여러 번 들어가면 csr 변환 시 자동으로 더해짐)
T = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                  shape=(len(catalog), len(tag_vocab)))
# TF-IDF: 모든 아티스트에 흔한 태그(예: '감성')의 비중은 낮추고, 드문 태그의 비중은 높이는 가중치
T = TfidfTransformer().fit_transform(T)
# TruncatedSVD: 희소 행렬을 128차원으로 압축 (함께 등장하는 태그끼리 같은 축으로 묶임)
# 참고: scikit-learn 문서 https://scikit-learn.org/stable/modules/decomposition.html#lsa
svd = TruncatedSVD(n_components=TAG_DIM, random_state=SEED)
tag_feat = svd.fit_transform(T)

def proportions(table):
    m = np.array([table[a] for a in catalog])
    return m / np.maximum(m.sum(axis=1, keepdims=True), 1)  # 합이 1이 되도록 (정보 없으면 0)

features = np.hstack([tag_feat, proportions(artist_genre), proportions(artist_era)]).astype(np.float32)

# ---------- 6단계: 저장 ----------
print("6/6 저장")
OUT.mkdir(exist_ok=True)
sp.save_npz(OUT / "interaction_matrix.npz", X)
np.save(OUT / "artist_features.npy", features)
np.save(OUT / "tag_embeddings.npy", svd.components_.T.astype(np.float32))  # 태그 → 128차원 (태그만 있는 입력용)


def dump(obj, name):
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


dump({"artist_ids": catalog,
      "artist_names": [artist_name[a] for a in catalog],
      "train_pids": [p["id"] for p in train],
      "tag_vocab": tag_vocab,
      "genre_codes": genre_codes}, "id_mappings.json")
dump(eval_sets, "split.json")
# 시연용: 곡 id → 카탈로그 열 번호 (카탈로그 아티스트가 있는 곡만). playlist_input.py에서 사용
dump({str(s): [col[a] for a in arts if a in col]
      for s, arts in song_artists.items() if any(a in col for a in arts)}, "song_to_artists.json")
dump([pl_count[a] for a in catalog], "artist_popularity.json")

def hist(values, edges):
    """edges [a, b, c] → 구간 a, b–(c-1), c 이상 별 개수 (보고서 그래프용)"""
    c = Counter(next(e for e in reversed(edges) if v >= e) for v in values)
    return {str(e): c[e] for e in edges}


stats = {
    "raw_playlists": len(raw_train),
    "removed_playlists": dict(removed),
    "cleaned_playlists": len(playlists),
    "train_artists_before_filter": len(pl_count),
    "artist_playlist_count_hist": hist(pl_count.values(), [1, 2, 5, 10, 20, 50, 100, 500, 1000]),
    "eval_catalog_artist_hist": hist([sum(a in col for a in p["share"]) for p in parts["dev"] + parts["test"]],
                                     [0, 1, 2, 3, 5, 10]),
    "train_tags_total": len(tag_count),
    "split_playlists": {k: len(v) for k, v in parts.items()},
    "eval_playlists_graded": {k: len(v) for k, v in eval_sets.items()},
    "catalog_artists": len(catalog),
    "matrix_shape": list(X.shape),
    "matrix_density": X.nnz / (X.shape[0] * X.shape[1]),
    "interactions_kept": len(vals) / sum(len(p["share"]) for p in train),
    "hidden_artist_coverage": coverage,
    "tag_vocab": len(tag_vocab),
    "tag_svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
    "artists_without_tags": int((T.getnnz(axis=1) == 0).sum()),
}
dump(stats, "stats.json")

with open(OUT / "feature_spec.md", "w", encoding="utf-8") as f:
    f.write(f"""# 전처리 산출물 명세 (preprocess.py, seed={SEED})

| 파일 | 내용 |
|---|---|
| interaction_matrix.npz | `scipy.sparse.load_npz`로 읽음. {X.shape[0]:,} × {X.shape[1]:,}. 행 = train 플레이리스트(`train_pids` 순서), 열 = 카탈로그 아티스트(`artist_ids` 순서), 값 = 그 플레이리스트에서 아티스트 곡이 차지하는 비율(song-share) |
| artist_features.npy | {features.shape[0]:,} × {features.shape[1]}. 행 순서는 interaction_matrix 열과 같음 |
| tag_embeddings.npy | {len(tag_vocab):,} × {TAG_DIM}. `tag_vocab` 순서. 태그만 있는 플레이리스트를 태그 공간으로 옮길 때 사용 |
| id_mappings.json | `artist_ids`, `artist_names`, `train_pids`, `tag_vocab`, `genre_codes` |
| split.json | `dev`, `test` 목록. 각 항목: `pid`, `input`(열 번호 → song-share), `answer`(숨긴 정답 중 카탈로그에 있는 열 번호), `answer_out_of_catalog`(카탈로그 밖이라 맞힐 수 없는 정답 수), `tags` |
| artist_popularity.json | 열 번호 순서의 train 등장 플레이리스트 수 (인기순 베이스라인용) |
| stats.json | 단계별 숫자 요약 (발표 자료용) |
| song_to_artists.json | 곡 id → 카탈로그 열 번호 목록. 시연 때 `playlist_input.py`가 곡 목록을 `input` 형식으로 바꿀 때 사용 |

## artist_features.npy 열 구성

| 열 | 블록 | 설명 |
|---|---|---|
| 0–{TAG_DIM - 1} | 태그 {TAG_DIM} | 아티스트가 등장한 train 플레이리스트 태그 → TF-IDF → TruncatedSVD. 태그가 전혀 없으면 0 |
| {TAG_DIM}–{TAG_DIM + 29} | 장르 30 | 대분류 장르 비율 (합 1). 순서는 `genre_codes` |
| {TAG_DIM + 30}–{TAG_DIM + 37} | 연대 8 | 발매연도 비율 (합 1): {', '.join(f'{lo}–{hi - 1}' for lo, hi in ERAS)} |

블록별 정규화·가중치는 모델 쪽에서 정함 (값의 크기가 블록마다 다름).

## 평가 규칙
- 플레이리스트 단위로 train/dev/test = 80/10/10, 카탈로그·태그·SVD는 train만으로 만듦
- dev/test에서 아티스트 {HIDE_RATIO:.0%}(최소 1명)를 숨김. 카탈로그 아티스트가 {MIN_EVAL_ARTISTS}명 미만인 플레이리스트는 채점에서 제외
- 추천 결과에서 `input`에 있는 아티스트는 빼고 Top-100으로 Recall@100, NDCG@100 계산
- 숨긴 정답이 모두 카탈로그 밖이라 `answer`가 빈 항목(dev {sum(not i['answer'] for i in eval_sets['dev'])}개, test {sum(not i['answer'] for i in eval_sets['test'])}개)은 채점에서 건너뜀
""")

# ---------- 자체 점검 ----------
assert features.shape == (len(catalog), TAG_DIM + 30 + len(ERAS))
assert not np.isnan(features).any()
train_ids = {p["id"] for p in train}
for name, items in eval_sets.items():
    for it in items:
        assert it["pid"] not in train_ids
        assert not set(map(int, it["input"])) & set(it["answer"])
print(json.dumps(stats, ensure_ascii=False, indent=2))
