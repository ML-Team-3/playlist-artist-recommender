"""
시연용: 곡 id 목록 → 모델 입력 형식 변환 (담당: 장승우)

split.json의 "input"과 같은 형식 {카탈로그 열 번호(str): song-share}를 만든다.
song-share = 그 아티스트 곡 수 / 플레이리스트 전체 곡 수 (preprocess.py와 같은 정의)

사용 예:
    from data.playlist_input import load_song_map, playlist_to_input
    song_map = load_song_map()
    query = playlist_to_input([525514, 129701, 383374], song_map)
    # query가 비어 있으면(카탈로그 아티스트 없음) 태그 또는 인기순으로 대체 추천

저장소 루트에서 실행하면(python data/playlist_input.py) 자체 점검 후 val.json 플레이리스트 하나를 변환해 보여준다.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_processed"


def load_song_map():
    with open(OUT / "song_to_artists.json", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def playlist_to_input(songs, song_map):
    if not songs:
        return {}
    cnt = Counter(c for s in songs for c in set(song_map.get(s, ())))  # 카탈로그 밖 곡은 무시
    return {str(c): round(n / len(songs), 4) for c, n in cnt.items()}


if __name__ == "__main__":
    import numpy as np
    import scipy.sparse as sp

    song_map = load_song_map()
    with open(OUT / "id_mappings.json", encoding="utf-8") as f:
        maps = json.load(f)
    raw = ROOT / "data_raw"

    # 자체 점검: train 플레이리스트를 변환한 결과가 interaction_matrix의 같은 행과 같아야 함
    X = sp.load_npz(OUT / "interaction_matrix.npz")
    row_of = {pid: i for i, pid in enumerate(maps["train_pids"])}
    with open(raw / "train.json", encoding="utf-8") as f:
        train = [p for p in json.load(f) if p["id"] in row_of][:500]
    for p in train:
        q = playlist_to_input(p["songs"], song_map)
        row = X[row_of[p["id"]]]
        assert set(map(int, q)) == set(row.indices.tolist())
        assert np.allclose([q[str(c)] for c in row.indices], row.data, atol=1e-4)
    print(f"자체 점검 통과: train 플레이리스트 {len(train)}개가 interaction_matrix와 일치")

    with open(raw / "val.json", encoding="utf-8") as f:
        p = next(p for p in json.load(f) if len(p["songs"]) >= 10)
    q = playlist_to_input(p["songs"], song_map)
    names = maps["artist_names"]
    top = sorted(q.items(), key=lambda kv: -kv[1])[:5]
    print(f"\nval 플레이리스트 {p['id']} '{p['plylst_title']}' ({len(p['songs'])}곡, 태그 {p['tags']})")
    print("카탈로그 아티스트", len(q), "명, 비중 상위:", [(names[int(c)], w) for c, w in top])
