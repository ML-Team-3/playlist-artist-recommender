# 공통 아티스트 추천 평가

Popularity, Content-Based + Popularity, Item-Based를 동일한 플레이리스트와 규칙으로 비교합니다. 모델·데이터를 변경하거나 학습하지 않습니다. 기본 실행은 Dev이며 Test는 명시적으로 선택해야 합니다.

## 설치 및 데이터 준비

Python 3.12를 권장합니다. 아래 명세는 Python 3.9–3.12용이며, 최종 검증 환경은 Python 3.12 / NumPy 2.0.2 / SciPy 1.13.1 / Matplotlib 3.9.4입니다. 프로젝트 루트에서 별도 가상환경을 사용하세요.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r evaluation/requirements.txt
python -B -m unittest evaluation.test_evaluation -v
```

테스트 15개는 합성 데이터를 임시 디렉터리에 생성하므로 실제 데이터 없이 실행할 수 있습니다. Matplotlib은 추가 실험 그래프와 관련 테스트에 필요합니다. 기본 평가에는 NumPy·SciPy만 필요합니다.

**전처리 산출물은 PR에 포함하지 않습니다.** 팀의 허가된 공유 경로에서 동일한 버전의 산출물을 받아 `data_processed/`에 배치하거나 `--data-dir`로 지정하세요. 데이터 재배포 권한은 팀에서 별도로 확인해야 합니다. 평가를 위해 전처리를 다시 실행할 필요는 없습니다.

| 필요한 파일 | 용도 |
|---|---|
| split.json | dev/test 목록과 pid, input, answer, answer_out_of_catalog |
| id_mappings.json | artist_ids 순서에 따른 카탈로그 식별 |
| artist_popularity.json | 카탈로그 열 순서의 Train 등장 플레이리스트 수 |
| artist_features.npy | 같은 아티스트 순서의 태그·장르·연대 특징 |
| interaction_matrix.npz | 학습 행렬 파일 식별용 해시; 평가 중 재학습하지 않음 |
| feature_spec.md | 전처리 명세 식별용 해시 |

기본 실행은 모델 선택과 관계없이 위 파일을 모두 식별합니다. 상세 데이터 명세는 팀에서 제공한 `feature_spec.md`를 참고하세요. `stats.json`, `tag_embeddings.npy`, `song_to_artists.json`은 평가에 필요하지 않습니다. `data/preprocess.py`, `data/playlist_input.py`도 import하거나 실행하지 않습니다.

Item-Based는 저장소의 `results/item_based/neighbors.npz`를 사용합니다. 체크포인트와 모든 산출물은 **동일한 아티스트 열 순서**를 사용해야 합니다. 크기가 같다고 의미적 매핑 일치가 증명되는 것은 아닙니다.

파일이 없으면 `FileNotFoundError`와 누락 경로가 출력됩니다(기본 실행은 `split.json`부터 읽음). 허가된 산출물을 받아 지정 디렉터리에 배치하세요. 자동 다운로드·전처리·재학습은 수행하지 않습니다. 패키지 누락 시 활성 가상환경에서 requirements를 설치하세요. Matplotlib 캐시 쓰기 경고는 `MPLCONFIGDIR`를 쓰기 가능한 디렉터리로 지정하여 해결할 수 있습니다.

## 기본 평가 실행

```bash
# 기본: Dev의 세 모델
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -B -m evaluation.run

# 경로·모델 명시
python -B -m evaluation.run --split dev --data-dir ./data_processed \
  --model-path ./results/item_based/neighbors.npz \
  --models popularity content_based item_based --output-dir ./results/evaluation

# Dev에서 모델 설정과 평가 규칙을 고정한 뒤 최종 Test 실행
python -B -m evaluation.run --split test
```

기본 경로는 프로젝트 루트 기준입니다. `--models`로 일부 모델만 선택할 수 있습니다. BLAS 스레드 제한은 작은 행렬 연산의 실행 비용을 줄이며 모델 계산식을 바꾸지 않습니다.

### 공통 평가 계약

- 빈 `answer`만 제외하고 모든 모델에 동일한 `input`·`answer`를 사용합니다. `input`의 아티스트 열 번호 → song-share를 그대로 전달합니다.
- Top-100을 한 번 생성하고 접두 목록으로 K=10/20/50/100을 계산합니다.
- Recall@K = 적중 수 / 카탈로그 정답 수. NDCG는 이진 정답·log2 순위 할인과 `min(K, 정답 수)` IDCG를 사용합니다. 플레이리스트별 지표의 macro average입니다.
- `answer_out_of_catalog`는 분모에서 제외합니다. 짧은 추천·빈 추천도 채점하며 부족한 순위는 미적중입니다.
- 순위 목록, 정수 ID(bool 제외), 범위, 중복, 입력 제외, 최대 길이를 공통 검증합니다. 위반 시 모델·pid를 포함한 오류를 내며 보정하지 않습니다.
- Popularity는 Train 등장 수 내림차순, 동점이면 열 번호 오름차순입니다.
- Content-Based는 기존 기본값(태그 1 / 장르 0.1 / 연대 0.2, 인기도 가산 0.3)을 확인합니다.
- Item-Based는 기존 로더·체크포인트를 사용하며 인기순 fallback을 추가하지 않습니다.
- 명세의 전체/빈 정답 제외/실제 채점 수는 Dev 10,257/71/10,186, Test 10,188/53/10,135입니다. 다른 개수면 경고하고 실측치를 기록합니다.

### 결과 저장 및 공유

매 실행마다 `results/evaluation/<split>_<UTC 시각>/`을 새로 생성하며 기존 결과를 덮어쓰지 않습니다. 출력 검증 실패 시 성공 결과를 저장하지 않습니다.

| 파일 | 내용 |
|---|---|
| summary.json | 지표·개수·설정·시간·환경·커밋·소스/데이터/체크포인트 SHA-256·경고 |
| comparison.csv | 모델별 지표·채점 수·평균 추천 수·100개 미만 비율·시간 |
| playlists.csv | 모델·pid별 지표·입력/정답/추천 개수·추천 순위 ID |

`playlists.csv`에는 원본 pid와 개별 추천이 포함되고 JSON에는 절대 경로 등 실행 환경 정보가 포함될 수 있습니다. **생성 디렉터리를 통째로 커밋하거나 공개 공유하지 마세요.** 집계표·그래프도 식별자·원본 목록 포함 여부를 검토해야 합니다. 이 PR에는 아래 집계 성능표만 문서에 포함하며 원본 결과 CSV/JSON·그래프는 포함하지 않습니다. `.gitignore`는 변경하지 않았으므로 명시적으로 파일을 Stage하세요.

해시는 파일 바이트를 식별하며 생성 이력이나 체크포인트 매핑 일치를 증명하지 않습니다. 학습 이웃 수 200은 기존 README 설명이며 체크포인트에서 관찰한 최대 저장 이웃 수와 구분하여 기록합니다.

`evaluation_seconds`는 추론·출력 검증·채점·집계, `load_seconds`는 모델 로딩 시간입니다. `total_seconds`는 로딩·해시·CSV 저장을 포함하고 마지막 요약 JSON 쓰기는 제외합니다.

## 추가 Dev 실험

`--experiments`와 로컬에서 생성한 세 모델의 전체 Dev 결과 디렉터리 `--reuse-results`를 명시합니다. 두 실험을 함께 또는 개별 실행할 수 있으며 Test는 거부합니다. 아래 `DEV_RUN`을 실제 디렉터리로 바꾸세요. 원본 결과 파일은 저장소에 제공하지 않습니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -B -m evaluation.run --experiments popularity-ablation input-length \
  --reuse-results results/evaluation/DEV_RUN

# 모델 추론 없는 길이별 집계
python -B -m evaluation.run --experiments input-length \
  --reuse-results results/evaluation/DEV_RUN
```

`--models`는 생략합니다(세 기본 모델 필요). 경로 옵션은 그대로 지원합니다. 기존 결과에는 생성 당시 절대 소스 경로가 기록되므로 다른 팀원의 결과 디렉터리를 직접 재사용하기보다 본인 체크아웃에서 기본 Dev를 실행한 뒤 재사용하세요.

재사용 전 데이터·체크포인트·모델·지표 코드 해시, 설정, 평가 수, 모델별 pid 집합, 출력 규칙을 검사합니다. 저장 추천의 지표를 재계산하여 원본 CSV·요약과 절대 오차 1e-12 이내 일치하는지도 확인합니다. 불일치·누락·중복 시 중단합니다.

- `popularity-ablation`: 같은 블록 가중치에서 공개 로더의 `popularity_weight=0`만 새로 추론하고 기본 0.3 결과를 재사용합니다. 계수 0에서도 `1e-6` 보정과 빈 입력의 인기순 대체 동작은 남습니다. 고정된 특징에서 가산항의 효과를 측정하며 콘텐츠 특징 자체의 효과를 분리한 실험은 아닙니다.
- `input-length`: `len(input)`으로 0 / 1–5 / 6–10 / 11–20 / 21+를 나눠 세 모델 결과를 집계합니다. 구간 합계와 가중평균이 전체 값과 일치하는지 검사합니다. 표본 수 1–29개는 경고 필드로 표시하고 구간을 바꾸지 않습니다. 빈 구간은 JSON null / CSV 빈칸이며 그래프에서 생략합니다. 정답 수·인기도·난이도 차이가 있으므로 길이 자체의 인과 효과로 해석하지 않습니다.

`results/evaluation/experiments_dev_<UTC 시각>/`에 `summary.json`, `popularity_ablation.csv`, `ablation_playlists.csv`, `input_length.csv`, PNG·SVG 그래프를 저장합니다. 새 설정도 기존 `evaluate_system()`의 출력 검증·지표 계산을 사용합니다.

## 집계 결과 및 검증

다음은 원본 pid·태그·입력·정답·추천 목록 없이 모델별 평균만 정리한 결과입니다. 개별 데이터를 재구성하는 데 필요한 목록은 포함하지 않습니다.

| 모델 | Dev Recall@100 | Dev NDCG@100 | Test Recall@100 | Test NDCG@100 |
|---|---:|---:|---:|---:|
| Popularity | 0.163491 | 0.079076 | 0.163144 | 0.077745 |
| Content-Based + Popularity | 0.482280 | 0.254337 | 0.485491 | 0.255580 |
| Item-Based | 0.563966 | 0.320038 | 0.562927 | 0.323300 |

Dev 10,186개, Test 10,135개를 채점했고 모든 K와 두 지표에서 동일한 모델 순위였습니다. 모두 평균 추천 수 100개, 100개 미만 비율 0%였습니다. 인기도 가산 제거 시 Dev Recall@100 0.436996, NDCG@100 0.222507이며 가산 후 상대 개선은 각각 10.36%, 14.31%였습니다. 입력 길이 네 구간 모두 Item-Based가 가장 높았습니다.

검증은 손계산 지표·출력 오류·구간 경계·캐시 손상 검사와 합성 데이터의 실제 로더/저장 통합 테스트 15개를 포함합니다. 실제 데이터 실행에서는 Dev 8개와 기존 `tune.evaluate()`의 일치를 확인하고 전체 저장 지표를 별도 공식으로 재계산했습니다. Test에서도 30,405개 추천 목록과 243,240개 지표 셀의 출력 규칙·집계를 확인했습니다. 모델·전처리 데이터는 변경하지 않았습니다.

NumPy/Accelerate에서 Content-Based `matmul`의 divide/overflow/invalid 경고가 발생하여 요약에 기록했습니다. 전체 Dev/Test의 프로필·내적 결과가 유한하고 `np.einsum(optimize=False)` 기준과 atol=2e-6, rtol=1e-5 이내 일치함을 별도 확인했습니다. 라이브러리 내부 경고 원인은 미확정입니다. Matplotlib/Pyparsing 사용 중단 예정 경고와 합성 테스트의 예상된 데이터 개수 경고도 발생할 수 있습니다.

이번 평가 개발 과정에서는 Test를 선택·튜닝에 사용하지 않았습니다. 사용자 설명상 PPT에는 과거 Test가 미사용으로 기록되어 있으나 문서 원문과 전체 개발 이력을 독립적으로 확인하지 못했습니다. **검증된 미사용 Test라고 단정하지 않습니다.** 결과는 현재 카탈로그 내 무작위 마스킹 아티스트 복원 성능이며 사용자 만족도·신규 아티스트·미래 시점 성능을 직접 입증하지 않습니다.
