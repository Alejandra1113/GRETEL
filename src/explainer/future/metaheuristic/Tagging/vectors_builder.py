import numpy as np
from typing import Callable, Literal, Optional

try:
    import networkx as nx
except ImportError:
    nx = None

Aggregator = Literal["sum", "product", "mean", "max", "min"]

class VectorsBuilder:
    n: int                    # number of nodes
    Eplus: int               # number of unordered pairs, n choose 2
    K: int                   # feature dimensions
    metrics: list[str]
    Matrix: np.ndarray       # (n, n) adjacency (assumed undirected, 0/1 or weights)
    X: np.ndarray            # (Eplus, K)

    def __init__(self, metrics: list[str], matrix: np.ndarray):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be a square (n, n) adjacency matrix")
        self.n = int(matrix.shape[0])
        self.Eplus = self.n * (self.n - 1) // 2
        self.K = len(metrics)
        self.metrics = metrics
        self.X = np.zeros((self.Eplus, self.K), dtype=np.float32)
        self.Matrix = matrix.astype(np.float32, copy=False)

        for dim, metric in enumerate(metrics):
            self.fill(dim=dim, metric=metric)

    # -------- public API --------

    def fill(self, dim: int, metric: str, **kwargs) -> None:
        if not (0 <= dim < self.K):
            raise ValueError(f"dim must be in [0, {self.K-1}]")

        m = metric.lower()
        if m in {"degree", "degree_centrality", "deg_centrality"}:
            self.fill_dimension_degree_centrality(dim=dim, **kwargs)
        elif m in {"closeness", "closeness_centrality"}:
            self.fill_dimension_closeness_centrality(dim=dim, **kwargs)
        elif m in {"eigenvector", "eigenvector_centrality"}:
            self.fill_dimension_eigenvector_centrality(dim=dim, **kwargs)
        elif m in {"betweenness", "betweenness_centrality"}:
            self.fill_dimension_betweenness_centrality(dim=dim, **kwargs)
        elif m in {"katz", "katz_centrality"}:
            self.fill_dimension_katz_centrality(dim=dim, **kwargs)
        elif m in {"pagerank", "page_rank"}:
            self.fill_dimension_pagerank(dim=dim, **kwargs)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # -------- centralities: implementers just produce node scores --------

    def fill_dimension_degree_centrality(
        self,
        dim: int,
        aggregator: Aggregator = "sum",
        normalized: bool = True,
    ) -> None:
        # degree centrality = degree / (n-1) if normalized else degree
        deg = self.Matrix.sum(axis=1)
        scores = (deg / (self.n - 1)) if normalized else deg
        self._fill_from_scores(dim, scores.astype(np.float32), aggregator)

    def fill_dimension_closeness_centrality(
        self,
        dim: int,
        aggregator: Aggregator = "sum",
        wf_improved: bool = True,
    ) -> None:
        if nx is not None:
            G = self._to_nx_graph()
            cc = nx.closeness_centrality(G, wf_improved=wf_improved)
            scores = np.array([cc[i] for i in range(self.n)], dtype=np.float32)
        else:
            # Unweighted closeness via BFS all-pairs shortest paths
            dist = self._all_pairs_shortest_path_lengths_unweighted()
            with np.errstate(divide="ignore", invalid="ignore"):
                # sum of distances to reachable nodes
                reachable = np.isfinite(dist)
                s = (dist * reachable).sum(axis=1)
                r = reachable.sum(axis=1)  # includes self
                # classic closeness with Wasserman-Faust improvement (if desired)
                if wf_improved:
                    # r-1 excludes self
                    scores = np.where(s > 0, (r - 1) / s * (r - 1) / (self.n - 1), 0.0)
                else:
                    scores = np.where(s > 0, (r - 1) / s, 0.0)
            scores = scores.astype(np.float32)

        self._fill_from_scores(dim, scores, aggregator)

    def fill_dimension_eigenvector_centrality(
        self,
        dim: int,
        aggregator: Aggregator = "sum",
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        # Power iteration on symmetric adjacency (handles weighted)
        A = self.Matrix
        x = np.ones(self.n, dtype=np.float64) / np.sqrt(self.n)
        for _ in range(max_iter):
            x_new = A @ x
            norm = np.linalg.norm(x_new)
            if norm == 0.0:  # empty graph
                break
            x_new /= norm
            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break
            x = x_new
        # Make nonnegative (can flip sign arbitrarily)
        if x.mean() < 0:
            x = -x
        scores = (x / x.max()) if x.max() > 0 else x
        self._fill_from_scores(dim, scores.astype(np.float32), aggregator)

    def fill_dimension_betweenness_centrality(
        self,
        dim: int,
        aggregator: Aggregator = "sum",
        normalized: bool = True,
        weight: Optional[str] = None,
    ) -> None:
        if nx is None:
            raise ImportError("betweenness_centrality requires networkx")
        G = self._to_nx_graph()
        bc = nx.betweenness_centrality(G, normalized=normalized, weight=None if weight is None else "weight")
        scores = np.array([bc[i] for i in range(self.n)], dtype=np.float32)
        self._fill_from_scores(dim, scores, aggregator)

    def fill_dimension_katz_centrality(
        self,
        dim: int,
        aggregator: Aggregator = "sum",
        alpha: Optional[float] = None,
        beta: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        if nx is None:
            raise ImportError("katz_centrality requires networkx")
        G = self._to_nx_graph()
        # If alpha not provided, pick a safe value: < 1/lambda_max
        if alpha is None:
            # quick spectral radius estimate via power iteration
            A = self.Matrix.astype(np.float64)
            x = np.ones(self.n) / np.sqrt(self.n)
            for _ in range(100):
                x = A @ x
                nrm = np.linalg.norm(x)
                if nrm == 0: break
                x /= nrm
            lam = np.linalg.norm(A @ x)  # Rayleigh estimate
            alpha = 0.85 / (lam + 1e-12) if lam > 0 else 0.1
        kc = nx.katz_centrality_numpy(G, alpha=alpha, beta=beta, normalized=True)
        scores = np.array([kc[i] for i in range(self.n)], dtype=np.float32)
        self._fill_from_scores(dim, scores, aggregator)

    def fill_dimension_pagerank(
        self,
        dim: int,
        aggregator: Aggregator = "sum",
        damping: float = 0.85,
        max_iter: int = 200,
        tol: float = 1.0e-06,
    ) -> None:
        if nx is None:
            raise ImportError("PageRank requires networkx")
        G = self._to_nx_graph()
        pr = nx.pagerank(G, alpha=damping, max_iter=max_iter, tol=tol)
        scores = np.array([pr[i] for i in range(self.n)], dtype=np.float32)
        self._fill_from_scores(dim, scores, aggregator)

    # -------- internal helpers --------

    def _fill_from_scores(self, dim: int, scores: np.ndarray, aggregator: Aggregator) -> None:
        """Given node scores (length n), fill X[:, dim] for all i<j using an aggregator."""
        if scores.shape != (self.n,):
            raise ValueError(f"scores must have shape ({self.n},)")
        agg = self._get_aggregator(aggregator)
        col = self.X[:, dim]  # view

        # Fill pairs ordered by i then j
        idx = 0
        for i in range(self.n - 1):
            si = scores[i]
            for j in range(i + 1, self.n):
                sj = scores[j]
                col[idx] = agg(float(si), float(sj))
                idx += 1

    def _to_nx_graph(self):
        """Undirected graph view of the adjacency matrix; weights kept if present."""
        if nx is None:
            raise ImportError("networkx not available")
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        A = self.Matrix
        # Add only upper triangle to avoid duplicate edges
        iu, ju = np.triu_indices(self.n, k=1)
        w = A[iu, ju]
        nz = np.nonzero(w)[0]
        edges = [(int(iu[k]), int(ju[k]), float(w[nz_idx])) for k, nz_idx in enumerate(nz)]
        # Faster: iterate over nz directly
        edges = [(int(iu[k]), int(ju[k]), float(w[k])) for k in nz]
        G.add_weighted_edges_from(edges)
        return G

    def _all_pairs_shortest_path_lengths_unweighted(self) -> np.ndarray:
        """BFS distances for an unweighted, undirected graph. inf if unreachable."""
        n = self.n
        A = (self.Matrix > 0).astype(np.uint8)
        dists = np.full((n, n), np.inf, dtype=np.float32)
        for s in range(n):
            dists[s, s] = 0.0
            # BFS queue
            q = [s]
            seen = np.zeros(n, dtype=bool)
            seen[s] = True
            while q:
                v = q.pop(0)
                # neighbors: A[v]==1
                nbrs = np.flatnonzero(A[v])
                for u in nbrs:
                    if not seen[u]:
                        seen[u] = True
                        dists[s, u] = dists[s, v] + 1.0
                        q.append(u)
        return dists

    @staticmethod
    def _pair_index(i: int, j: int, n: int) -> int:
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
