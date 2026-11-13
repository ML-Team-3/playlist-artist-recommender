"""Item-based collaborative filtering over artist columns."""

import numpy as np
from scipy import sparse


def fit_neighbors(interactions, k, block_size=128):
    """Fit a sparse directed graph with up to k cosine neighbors per artist."""
    artist_count = interactions.shape[1]
    if not 1 <= k < artist_count:
        raise ValueError("k must be between 1 and the number of artists minus 1")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    interactions = sparse.csr_matrix(interactions, dtype=np.float32)
    norms = np.sqrt(np.asarray(interactions.power(2).sum(axis=0)).ravel())
    inverse_norms = np.divide(1, norms, out=np.zeros_like(norms), where=norms > 0)
    artists = interactions.multiply(inverse_norms).T.tocsr()
    playlists = artists.T.tocsr()

    neighbors = np.empty((artist_count, k), dtype=np.int32)
    similarities = np.empty((artist_count, k), dtype=np.float32)
    for start in range(0, artist_count, block_size):
        stop = min(start + block_size, artist_count)
        block = (artists[start:stop] @ playlists).toarray()
        rows = np.arange(stop - start)
        block[rows, start + rows] = 0
        # Stable ordering resolves equal cosine scores by artist column.
        order = np.argsort(-block, axis=1, kind="stable")[:, :k]
        neighbors[start:stop] = order
        similarities[start:stop] = block[rows[:, None], order]

    return neighbors, similarities


def neighbor_graph(neighbors, similarities, k):
    """Use the first k ranked neighbors from a fitted neighborhood."""
    artist_count, max_k = neighbors.shape
    if not 1 <= k <= max_k:
        raise ValueError("k must be between 1 and the fitted neighborhood size")
    graph = sparse.csr_matrix(
        (
            similarities[:, :k].flatten(),
            neighbors[:, :k].flatten(),
            np.arange(artist_count + 1) * k,
        ),
        shape=(artist_count, artist_count),
    )
    graph.eliminate_zeros()
    return graph


def recommend(graph, artist_weights, limit=100, popularity_order=None):
    """Rank unseen artist columns for a mapping of column to song-share."""
    if not artist_weights:
        if popularity_order is None:
            return np.empty(0, dtype=np.intp)
        return popularity_order[:limit]

    columns = np.fromiter(artist_weights, dtype=np.intp)
    if np.any(columns < 0) or np.any(columns >= graph.shape[0]):
        raise ValueError("input contains an artist column outside the catalog")
    weights = np.fromiter(artist_weights.values(), dtype=graph.dtype)
    query = sparse.csr_matrix(
        (weights, (np.zeros_like(columns), columns)), shape=(1, graph.shape[0])
    )
    scores = (query @ graph).toarray().ravel()
    scores[columns] = 0
    candidates = np.flatnonzero(scores > 0)
    return candidates[np.argsort(-scores[candidates], kind="stable")[:limit]]


def load_recommender(model_path):
    """Load the fitted graph once and return a callable for split.json inputs."""
    graph = sparse.load_npz(model_path)

    def recommend_playlist(playlist_input, top_n=100):
        if top_n < 0:
            raise ValueError("top_n must be non-negative")
        return recommend(graph, playlist_input, limit=top_n).tolist()

    return recommend_playlist
