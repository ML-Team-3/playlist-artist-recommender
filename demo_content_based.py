"""
Content-based 모델 확인용 스크립트 (담당: 김택현)

1) Dev 첫 번째 플레이리스트의 입력·태그·추천 결과를 아티스트 이름으로 출력
2) Dev 전체 입력에 대해 출력 형식이 팀 규칙에 맞는지 검사
   (100개, 정수, 중복 없음, 입력 아티스트 제외)

실행: python demo_content_based.py
      python demo_content_based.py --data-dir "../전처리 데이터/data_processed"
"""

import argparse
import json
import time
from pathlib import Path

from models.content_based import load_recommender


def main():
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo.parent / "전처리 데이터" / "data_processed")
    args = parser.parse_args()

    with (args.data_dir / "split.json").open(encoding="utf-8") as f:
        dev = json.load(f)["dev"]
    with (args.data_dir / "id_mappings.json").open(encoding="utf-8") as f:
        names = json.load(f)["artist_names"]

    recommend = load_recommender(args.data_dir)

    # 1) 첫 번째 Dev 플레이리스트 예시
    p = dev[0]
    print("Content-based 아티스트 추천")
    print("태그:", p["tags"])
    print("입력 아티스트:", [names[int(c)] for c in p["input"]])
    result = recommend(p["input"])
    answer = set(p["answer"])
    for rank, col in enumerate(result[:20], 1):
        mark = "  <- 정답" if col in answer else ""
        print(f"{rank}. {names[col]}{mark}")
    print(f"숨긴 정답 {len(answer)}명 중 Top 100 안에 {len(answer & set(result))}명")

    # 2) Dev 전체 형식 검사
    start, problems = time.time(), 0
    for p in dev:
        given = {int(k) for k in p["input"]}
        out = recommend(p["input"], top_n=100)
        ok = (len(out) == 100 and len(set(out)) == 100
              and all(type(c) is int for c in out) and not given & set(out))
        problems += not ok
    print(f"\nDev {len(dev):,}개 검사 완료: 형식 오류 {problems}개, {time.time() - start:.1f}초")


if __name__ == "__main__":
    main()
