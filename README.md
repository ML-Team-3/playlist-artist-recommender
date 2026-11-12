# Playlist-Aware Artist Recommendation

Machine Learning Term Project | Team 03 | 2026

This project aims to recommend artists based on playlists using the Kakao Arena Melon Playlist Dataset. It will compare Item-Based Collaborative Filtering and Content-Based Filtering.

| Team Member | Role |
| --- | --- |
| 장승우 | Data Research & Preprocessing |
| 김택현, 김진서 | Model Implementation |
| 박현서 | Evaluation & Validation |

## Item-based collaborative filtering

Model code lives in `models/item_based/`, and the trained model lives in `results/item_based/`. The content-based model can use `models/content_based/` and `results/content_based/`.

Run from this repository with Python 3, NumPy, and SciPy. Point `--data-dir` to the directory containing the training matrix `interaction_matrix.npz`. Set `--neighbors` to change the number of cosine neighbors kept per artist. The default is 200.

```bash
python3 -m models.item_based.train --data-dir "../전처리 데이터/data_processed" --neighbors 200
```

For evaluation, load the pretrained model once and pass each playlist's `input` dictionary from `split.json` directly to the returned function. The checkpoint was trained with K=200 using the train matrix. The returned Python `list[int]` contains ranked column indices corresponding to `artist_names` in the matching `id_mappings.json`.

```python
from models.item_based import load_recommender

recommend = load_recommender("results/item_based/neighbors.npz")
artist_columns = recommend(playlist["input"], top_n=100)
```

The ranker excludes input artists and duplicates. It returns up to `top_n` positive-scoring candidates, and returns an empty list for an empty input. Use dev to compare K values and reserve test for final evaluation.

Run `demo_item_based.py` to print recommendations for the first dev playlist using the saved model.
