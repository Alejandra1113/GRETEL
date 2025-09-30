import numpy as np
from typing import Iterable, Optional, Callable, Literal, Union

try:
    import networkx as nx
except ImportError:
    nx = None

Aggregator = Literal["sum", "product", "mean", "max", "min"]

class VectorsBuilder:
    Eplus: int   # number of possible edges (n choose 2)
    K: int       # number of dimensions
    metrics: list[str] # list of metrics, length K
    Matrix: np.ndarray  # adjacency matrix, shape (n,n)
    X: np.ndarray       # output matrix, shape (Eplus, K)

    def __init__(self, k: int, metrics: list[str], matrix: np.ndarray):
        self.Eplus = matrix.shape[0]
        self.K = k
        self.metrics = metrics
        self.X = np.zeros((self.Eplus, self.K), dtype=np.float32)
        self.Matrix = matrix
        for dim, metric in enumerate(metrics):
            self.fill(dim=dim, metric=metric)
        
    
    def fill(self, dim: int, metric: str, **kwargs) -> None:
        if "degree" is metric:
            raise NotImplementedError("raw degree not implemented yet")
        if "degree_centrality" is metric or "deg_centrality" is metric:
            self.fill_dimension_degree_centrality(dim=dim, **kwargs)
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
    def fill_dimension_degree_centrality(
        self,
        dim: int,
        aggregator: Aggregator = "sum",
    ) -> None:
        """
        Fill X[:, dim] for all unordered pairs (i, j) with an aggregation of
        degree centrality for nodes i and j.

        Parameters
        ----------
        dim : int
            Feature dimension/column in X to fill (0 <= dim < K).
        aggregator : {'sum','product','mean','max','min'}
            How to combine centrality(i) and centrality(j). Default 'sum'.
        """
        if not (0 <= dim < self.K):
            raise ValueError(f"dim must be in [0, {self.K-1}]")

        agg = self._get_aggregator(aggregator)

        if nodes_order is None:
            nodes_order = list(range(self.n))


        if self.Matrix.shape != (self.n, self.n):
            raise ValueError(f"adjacency must be shape ({self.n}, {self.n})")
        # Degree centrality = degree / (n-1)
        deg = self.Matrix.sum(axis=1)
        dc = (deg / (self.n - 1)).astype(np.float32)

        # Fill the column for every pair (i, j) with i<j
        col = self.X[:, dim]  # view

        for i in range(self.n - 1):
            ci = dc[i]
            base = i * (2 * self.n - i - 1) // 2  # offset for row start at this i
            for j in range(i + 1, self.n):
                cj = dc[j]
                row = base + (j - i - 1)
                col[row] = agg(ci, cj)
        
            
# -------- internal helpers --------

    @staticmethod
    def _pair_index(i: int, j: int, n: int) -> int:
        """
        Deterministic index for 0 <= i < j < n, ordered by i then j.
        Formula = triangular number offset for i + (j - i - 1)
        offset(i) = i * (2n - i - 1) // 2
        """
        if not (0 <= i < j < n):
            raise ValueError("Require 0 <= i < j < n")
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    @staticmethod
    def _get_aggregator(name: Aggregator) -> Callable[[float, float], float]:
        if name == "sum":
            return lambda a, b: a + b
        if name == "product":
            return lambda a, b: a * b
        if name == "mean":
            return lambda a, b: 0.5 * (a + b)
        if name == "max":
            return lambda a, b: a if a >= b else b
        if name == "min":
            return lambda a, b: a if a <= b else b
        raise ValueError(f"Unknown aggregator: {name}")

    

