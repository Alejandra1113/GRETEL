import hnswlib
import numpy as np
from typing import Iterable, Optional

class ANNIndexWeighted:
    """
    HNSW ANN index over weighted-cosine space.
    """

    def __init__(
        self,
        X: np.ndarray,                 # shape (N, K), float32 preferred
        ef_construction: int = 100,
        ef: int = 100,
        initial_weight: float = 0.5,   # start all dims at 0.5
        seed: Optional[int] = None
    ):
        assert X.ndim == 2, "X must be (N,K)"
        self.X_raw = X.astype(np.float32, copy=True)
        self.N, self.K = self.X_raw.shape   # pairs of nodes, each with K-dim vector

        # weights in [0,1], start at 0.5
        self.w = np.full(self.K, float(initial_weight), dtype=np.float32)
        self.ef_construction = ef_construction
        self.ef = ef
        self.seed = seed

        # build first index
        self._rebuild_index()

    # -------- public API --------

    def set_weights(self, w: np.ndarray):
        """Set full weight vector (shape K), values in [0,1], then rebuild."""
        w = np.asarray(w, dtype=np.float32)
        assert w.shape == (self.K,), "w must have shape (K,)"
        # clip to [0,1]
        np.clip(w, 0.0, 1.0, out=w)
        self.w = w
        self._rebuild_index()
    
    def recalculate_weights_from_indices(
        self,
        idxs: Iterable[int],
        old_weight_share: float = 0.5,  # 50% current weights
    ) -> None:
        """
        Update self.w using the current weights (old_weight_share)
        and per-dimension signal from X_raw rows in `idxs` (1 - old_weight_share).
        Higher per-dim values in X_raw -> higher new weight.
        Always keeps weights in [0,1], then rebuilds the index.

        Strategy:
          score_j = mean(abs(X_raw[idxs, j]))    # per-dimension magnitude
          scaled_j = score_j / max(score)        # in [0,1]
          new_w = old_weight_share * w + (1-old_weight_share) * scaled
        """
        idxs = np.asarray(list(idxs), dtype=np.int32)
        if idxs.size == 0:
            return  # nothing to do

        if np.any((idxs < 0) | (idxs >= self.N)):
            raise IndexError("Some indices in idxs are out of bounds.")

        # Per-dimension signal from the selected rows
        Xsel = self.X_raw[idxs]                # (m, K)
        score = np.mean(np.abs(Xsel), axis=0)  # (K,)

        # Normalize to [0,1] (avoid divide-by-zero)
        maxv = np.max(score)
        if maxv <= 1e-12:
            scaled = np.zeros_like(score, dtype=np.float32)
        else:
            scaled = (score / maxv).astype(np.float32)

        # Blend 50/50 with existing weights (clip to [0,1])
        alpha = float(old_weight_share)
        alpha = min(1.0, max(0.0, alpha))
        new_w = alpha * self.w + (1.0 - alpha) * scaled
        new_w = np.clip(new_w, 0.0, 1.0).astype(np.float32)

        # Set & rebuild to reflect the new distance metric
        self.w = new_w
        self._rebuild_index()
        
        
    def update_weights(self, idxs: Iterable[int], vals: Iterable[float]):
        """Update some dimensions and rebuild. idxs are dim indices [0..K-1]."""
        for i, v in zip(idxs, vals):
            if 0 <= i < self.K:
                self.w[i] = np.float32(min(1.0, max(0.0, v)))
            else:
                raise IndexError(f"dimension {i} out of range [0,{self.K-1}]")
        self._rebuild_index()

    def get_weights(self) -> np.ndarray:
        return self.w.copy()

    
    # -------- Adding --------
    def neighbors_of_S_union(
        self,
        S: set,
        top_k: int
    ) -> list[int]:
        """
        For each s in S, retrieve its neighbors in the weighted-cosine space,
        union all neighbor labels (deduped), then return a RANDOM batch of size
        `top_k` from that union (excluding S itself).

        If fewer than `top_k` unique candidates exist, returns all of them and raises a warning.
        """

        S_arr = np.fromiter(S, dtype=np.int32)

        # Ask a bit more per query to offset filtering of S itself
        per_query_k = min(self.N, top_k + len(S))
        labels_union = set()

        for s in S_arr:
            # weighted + normalized query vector
            q = self._normalize(self._weighted(self.X_raw[s]))
            local_labels, _ = self.index.knn_query(q, k=per_query_k)
            # Deduplicate across queries and exclude S itself
            for lab in local_labels[0]:
                if lab not in S:
                    labels_union.add(int(lab))

        # Random sample of size top_k (or all if fewer available)
        if not labels_union:
            return []
        candidates = np.fromiter(labels_union, dtype=np.int32)
        k = min(top_k, candidates.size)

        # np.random.choice without replacement over a 1D array
        picked = np.random.choice(candidates, size=k, replace=False)
        assert len(picked) == top_k, "picked fewer than top_k"
        return picked.tolist()

    def neighbors_of_centroid(
        self,
        S: list,
        top_k : int,
    ) -> list[int]:
        """
        Find neighbors near the weighted centroid of rows in S.
        Returns list of indices (and optional distances) in 0-based dataset coords.
        """
        S = np.asarray(S, dtype=np.int32)
        top_k_per_query = min(self.N, top_k + len(S))
        # Weighted & normalized query
        centroid = self._weighted(self.X_raw[S]).mean(axis=0)
        centroid = self._normalize(centroid)

        labels, _ = self.index.knn_query(centroid, k=top_k_per_query)
        result = labels[0]

        result = [r for r in result if r not in S]
        result = result[:top_k]

        return result
    
    
    
    # -------- Removing --------
    
    def prune_farthest_in_S(
        self,
        S: set,                 # set of 0-based dataset indices
        remove_k: int,          # how many to drop
        temperature: float = 0.3,   # lower = greedier, 0 -> deterministic farthest
        rng: np.random.Generator | None = None,
    ) -> tuple[list[int], list[int]]:
        """
        Stochastically remove `remove_k` elements of S, biased toward those farthest
        from S's weighted centroid (cosine distance in weighted space).

        - `temperature=0`  -> deterministic: drop the farthest `remove_k`.
        - higher temperature -> more randomness.
        Returns (kept_indices, removed_indices), both as lists of dataset indices.
        """
        S = np.asarray(list(S), dtype=np.int32)
        if remove_k <= 0:
            return S.tolist(), []
        if remove_k >= S.size:
            return [], S.tolist()

        if rng is None:
            rng = np.random.default_rng()

        # Weighted + normalized reps of S
        Xs = self._normalize(self._weighted(self.X_raw[S]))

        # Weighted centroid (normalized)
        centroid = self._normalize(Xs.mean(axis=0))

        # Cosine distance to centroid: d = 1 - cos_sim
        sims = Xs @ centroid
        dists = 1.0 - sims  # in [0,2] for normalized vectors

        # Deterministic fallback (temperature == 0)
        if temperature <= 0.0:
            idx = np.argpartition(dists, -remove_k)[-remove_k:]
            idx = idx[np.argsort(dists[idx])[::-1]]
            mask = np.ones(S.size, dtype=bool)
            mask[idx] = False
            return S[mask].tolist(), S[idx].tolist()

        # Probabilistic removal via softmax over distances
        # More distant => larger probability
        eps = 1e-12
        logits = dists / max(temperature, eps)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs_sum = probs.sum()
        if probs_sum <= 0 or not np.isfinite(probs_sum):
            probs = np.full_like(probs, 1.0 / probs.size)
        else:
            probs = probs / probs_sum

        # Sample without replacement according to probs
        idx_remove = rng.choice(S.size, size=remove_k, replace=False, p=probs)
        mask = np.ones(S.size, dtype=bool)
        mask[idx_remove] = False

        kept = S[mask].tolist()
        removed = S[idx_remove].tolist()
        return kept, removed

    
    
    
    # -------- internals --------
    def _weighted(self, X: np.ndarray) -> np.ndarray:
        """Apply per-dim weights: w ⊙ X. Accepts (K,) or (N,K)."""
        return X * self.w

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        eps = 1e-12
        if X.ndim == 1:
            return X / (np.linalg.norm(X) + eps)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

    def _rebuild_index(self) -> None:
        """Rebuild HNSW on weighted+normalized data (distance definition changes)."""
        Xw = self._normalize(self._weighted(self.X_raw)).astype(np.float32, copy=False)

        self.index = hnswlib.Index(space="cosine", dim=self.K)
        self.index.init_index(
            max_elements=self.N,
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=(self.seed if self.seed is not None else 100),
        )
        labels = np.arange(self.N, dtype=np.int32)
        self.index.add_items(Xw, labels)
        self.index.set_ef(self.ef)