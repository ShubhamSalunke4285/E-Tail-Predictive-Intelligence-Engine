"""Item-based collaborative filtering recommender.

Builds a sparse user x item matrix from the implicit-feedback ratings, computes
item-item cosine similarity, and scores candidate items for each user as the
similarity-weighted sum of the items they have already engaged with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

import config


class ItemBasedCF:
    def __init__(self, top_n: int = config.CF_TOP_N):
        self.top_n = top_n
        self.user_index: dict[int, int] = {}
        self.item_index: dict[int, int] = {}
        self.items: np.ndarray | None = None
        self.matrix: csr_matrix | None = None
        self.item_sim: np.ndarray | None = None

    def fit(self, ratings: pd.DataFrame) -> "ItemBasedCF":
        users = ratings["user_id"].unique()
        items = ratings["product_id"].unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {p: i for i, p in enumerate(items)}
        self.items = items

        rows = ratings["user_id"].map(self.user_index).to_numpy()
        cols = ratings["product_id"].map(self.item_index).to_numpy()
        vals = ratings["rating"].to_numpy(dtype=float)

        self.matrix = csr_matrix(
            (vals, (rows, cols)), shape=(len(users), len(items))
        )
        # Item-item similarity (items are columns -> transpose).
        self.item_sim = cosine_similarity(self.matrix.T, dense_output=True)
        np.fill_diagonal(self.item_sim, 0.0)
        return self

    def recommend_all(self) -> pd.DataFrame:
        """Return top-N recommendations for every user as a tidy DataFrame."""
        # scores[user, item] = sum over engaged items of similarity. A single
        # sparse matmul does this for all users at once.
        scores = self.matrix @ self.item_sim                # (n_users, n_items)
        scores = np.asarray(scores)

        # Mask items the user already interacted with.
        already = self.matrix.toarray() > 0
        scores[already] = -np.inf

        inv_user = {i: u for u, i in self.user_index.items()}
        records = []
        top_n = min(self.top_n, scores.shape[1])
        # argpartition for the top-N, then sort just those.
        top_idx = np.argpartition(-scores, top_n - 1, axis=1)[:, :top_n]
        for row in range(scores.shape[0]):
            cand = top_idx[row]
            cand = cand[np.argsort(-scores[row, cand])]
            user_id = inv_user[row]
            for rank, item_col in enumerate(cand, start=1):
                s = scores[row, item_col]
                if not np.isfinite(s) or s <= 0:
                    continue
                records.append((user_id, int(self.items[item_col]), rank,
                                round(float(s), 4)))

        return pd.DataFrame(
            records, columns=["user_id", "product_id", "rank", "score"]
        )

    def item_neighbors(self, top_k: int = 20) -> pd.DataFrame:
        """Top-K most similar items per item (the 'customers also bought' map).

        Stored once so the storefront can build fresh, session-based
        recommendations at request time without retraining the whole model.
        """
        sim = self.item_sim
        k = min(top_k, sim.shape[1] - 1)
        top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        records = []
        for row in range(sim.shape[0]):
            cols = top[row][np.argsort(-sim[row, top[row]])]
            pid = int(self.items[row])
            for c in cols:
                s = float(sim[row, c])
                if s <= 0:
                    continue
                records.append((pid, int(self.items[c]), round(s, 4)))
        return pd.DataFrame(records,
                            columns=["product_id", "similar_id", "score"])


def train_and_score(ratings: pd.DataFrame):
    """Fit the model and return (recommendations, item_neighbors)."""
    # Filter ultra-sparse users so similarity scores are meaningful.
    counts = ratings.groupby("user_id").size()
    keep = counts[counts >= config.CF_MIN_INTERACTIONS].index
    filtered = ratings[ratings["user_id"].isin(keep)]
    print(f"  CF training on {filtered['user_id'].nunique():,} users / "
          f"{filtered['product_id'].nunique():,} items")

    model = ItemBasedCF().fit(filtered)
    recs = model.recommend_all()
    neighbors = model.item_neighbors(top_k=20)
    print(f"  generated {len(recs):,} recommendations, "
          f"{len(neighbors):,} item-similarity links")
    return recs, neighbors
