import heapq
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, List, Set, Tuple, Dict, Iterator

# ------- Your LRU-ish cache (unchanged) -------
class FixedSizeCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def add(self, item: Iterable[int]):
        key = frozenset(item)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = None
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def contains(self, item: Iterable[int]) -> bool:
        return frozenset(item) in self.cache


@dataclass
class SimpleTagger:
    EPlus: int

    # Diversity/greediness knobs
    rcl_size_add: int = 8
    rcl_size_remove: int = 8

    # Removal scoring
    removal_metric: str = "nn_gap"  # or "value"

    # Soft action-specific cooldowns
    add_cooldown: int = 10
    remove_cooldown: int = 10
    add_penalty_weight: float = 0.75
    remove_penalty_weight: float = 0.75

    # Retry limit for skipping cached duplicates
    max_skip_tries: int = 5_000

    rng: random.Random = field(default_factory=random.Random)

    # Global memory
    solution_cache: FixedSizeCache = field(default_factory=lambda: FixedSizeCache(2000))
    _iter: int = 0
    _last_added_iter: Dict[int, int] = field(default_factory=dict)
    _last_removed_iter: Dict[int, int] = field(default_factory=dict)

    # Per-baseline reusable state
    _add_frontiers: Dict[frozenset, "AdditionFrontier"] = field(default_factory=dict)
    _removal_planners: Dict[frozenset, "RemovalPlanner"] = field(default_factory=dict)

    # ------------------ PUBLIC: acceptance hook ------------------
    def accept(self, new_solution: Set[int], prev_solution: Set[int] | None = None):
        """
        Call this when you decide a yielded candidate is GOOD and becomes your current baseline.
        This:
          - bumps iteration,
          - updates cooldown memories for elements actually added/removed,
          - rebinds per-baseline frontiers to the new baseline.
        If you pass prev_solution, we infer deltas; else we just record the solution.
        """
        self._iter += 1
        if prev_solution is not None:
            added = set(new_solution) - set(prev_solution)
            removed = set(prev_solution) - set(new_solution)
            for x in added:
                self._last_added_iter[x] = self._iter
            for x in removed:
                self._last_removed_iter[x] = self._iter

        self._rebind_frontiers(new_baseline=new_solution)

    # ------------------ GENERATORS ------------------
    def gen_additions(self, base_solution: Set[int], ks: List[int], n: int) -> Iterator[Set[int]]:
        """
        Yield n candidate solutions made by adding k elements; k cycles over ks.
        Reuses a persistent frontier keyed by the BASE solution only.
        Does NOT mutate your base solution nor cooldowns; use accept() when you keep one.
        """
        assert ks, "ks must be a non-empty list"
        baseline_key = frozenset(base_solution)
        frontier = self._add_frontiers.get(baseline_key)
        if frontier is None:
            frontier = AdditionFrontier(
                EPlus=self.EPlus,
                base=set(base_solution),
                add_penalty=lambda x: self._add_penalty(x),
                rcl_size=self.rcl_size_add,
                rng=self.rng,
            )
            self._add_frontiers[baseline_key] = frontier

        yielded = set()  # avoid dup within this call
        i = 0
        tries = 0
        while len(yielded) < n and tries < self.max_skip_tries:
            k = ks[i % len(ks)]
            cand_added = frontier.pick_k(k)  # list[int]
            # Candidate solution = base ∪ picked
            candidate = set(base_solution)
            for x in cand_added:
                if x not in candidate:
                    candidate.add(x)

            key = frozenset(candidate)
            if key not in yielded and not self.solution_cache.contains(candidate):
                yielded.add(key)
                self.solution_cache.add(candidate)
                yield candidate
            else:
                tries += 1
            i += 1

    def gen_removals(self, base_solution: Set[int], ks: List[int], n: int) -> Iterator[Set[int]]:
        """
        Yield n candidate solutions made by removing k elements; k cycles over ks.
        Uses a persistent removal planner keyed by the BASE solution only.
        """
        assert ks, "ks must be a non-empty list"
        baseline_key = frozenset(base_solution)
        planner = self._removal_planners.get(baseline_key)
        if planner is None:
            planner = RemovalPlanner(
                base=set(base_solution),
                metric=self.removal_metric,
                rcl_size=self.rcl_size_remove,
                remove_penalty=lambda x: self._remove_penalty(x),
                rng=self.rng,
            )
            self._removal_planners[baseline_key] = planner

        yielded = set()
        i = 0
        tries = 0
        while len(yielded) < n and tries < self.max_skip_tries:
            k = min(ks[i % len(ks)], len(base_solution))
            to_remove = planner.pick_k(k)
            candidate = set(base_solution)
            for x in to_remove:
                candidate.discard(x)

            key = frozenset(candidate)
            if key not in yielded and not self.solution_cache.contains(candidate):
                yielded.add(key)
                self.solution_cache.add(candidate)
                yield candidate
            else:
                tries += 1
            i += 1

    def gen_swaps(self, base_solution: Set[int], ks: List[int], n: int) -> Iterator[Set[int]]:
        """
        Yield n candidate solutions by removing k then adding k (k cycles over ks).
        We reuse the removal planner; for addition we spin a short-lived frontier per proposal
        seeded from the post-removal state. (Keeps quality without building the universe.)
        """
        assert ks, "ks must be a non-empty list"
        baseline_key = frozenset(base_solution)
        planner = self._removal_planners.get(baseline_key)
        if planner is None:
            planner = RemovalPlanner(
                base=set(base_solution),
                metric=self.removal_metric,
                rcl_size=self.rcl_size_remove,
                remove_penalty=lambda x: self._remove_penalty(x),
                rng=self.rng,
            )
            self._removal_planners[baseline_key] = planner

        yielded = set()
        i = 0
        tries = 0
        while len(yielded) < n and tries < self.max_skip_tries:
            k = min(ks[i % len(ks)], len(base_solution))
            # Remove k via planner
            to_remove = planner.pick_k(k)
            mid = set(base_solution)
            for x in to_remove:
                mid.discard(x)

            # Add k via a local frontier built from 'mid'
            add_frontier = AdditionFrontier(
                EPlus=self.EPlus,
                base=mid,
                add_penalty=lambda x: self._add_penalty(x),
                rcl_size=self.rcl_size_add,
                rng=self.rng,
            )
            to_add = add_frontier.pick_k(k)
            candidate = set(mid)
            for x in to_add:
                candidate.add(x)

            key = frozenset(candidate)
            if key not in yielded and not self.solution_cache.contains(candidate):
                yielded.add(key)
                self.solution_cache.add(candidate)
                yield candidate
            else:
                tries += 1
            i += 1

    # ------------------ INTERNAL: cooldown penalties ------------------
    def _add_penalty(self, x: int) -> float:
        t = self._last_added_iter.get(x)
        if t is None or self.add_cooldown <= 0:
            return 0.0
        dt = max(0, self._iter - t)
        if dt >= self.add_cooldown:
            return 0.0
        return self.add_penalty_weight * (1.0 - dt / self.add_cooldown)

    def _remove_penalty(self, x: int) -> float:
        t = self._last_removed_iter.get(x)
        if t is None or self.remove_cooldown <= 0:
            return 0.0
        dt = max(0, self._iter - t)
        if dt >= self.remove_cooldown:
            return 0.0
        return self.remove_penalty_weight * (1.0 - dt / self.remove_cooldown)

    # ------------------ INTERNAL: rebind per-baseline state ------------------
    def _rebind_frontiers(self, new_baseline: Set[int]):
        """
        When you accept a candidate, rebase the stateful planners/frontiers to that solution.
        """
        self._add_frontiers.clear()
        self._removal_planners.clear()
        key = frozenset(new_baseline)
        self._add_frontiers[key] = AdditionFrontier(
            EPlus=self.EPlus,
            base=set(new_baseline),
            add_penalty=lambda x: self._add_penalty(x),
            rcl_size=self.rcl_size_add,
            rng=self.rng,
        )
        self._removal_planners[key] = RemovalPlanner(
            base=set(new_baseline),
            metric=self.removal_metric,
            rcl_size=self.rcl_size_remove,
            remove_penalty=lambda x: self._remove_penalty(x),
            rng=self.rng,
        )


# --------- Helpers: Addition frontier with reusable heap ---------
class AdditionFrontier:
    """
    Maintains a layered min-heap over the number line starting from the current base set.
    Key = (dist + soft_penalty, dist) so we pop closest first but respect cooldown.
    Never materializes the universe; validates with bounds and base-membership.
    """
    __slots__ = ("EPlus", "base", "add_penalty", "rcl_size", "rng", "heap", "seen")

    def __init__(self, EPlus: int, base: Set[int], add_penalty, rcl_size: int, rng: random.Random):
        self.EPlus = EPlus
        self.base = set(base)
        self.add_penalty = add_penalty
        self.rcl_size = rcl_size
        self.rng = rng
        self.heap: List[Tuple[float, int, int]] = []  # (key, dist, x)
        self.seen: Set[int] = set(self.base)
        self._seed_initial_neighbors()

    def pick_k(self, k: int) -> List[int]:
        """
        Pick k additions (not in base). Reuses/extends the heap across calls.
        Note: selecting items also grows the frontier; next calls will be faster/richer.
        """
        picked: List[int] = []
        while len(picked) < k and self.heap:
            key, dist, x = heapq.heappop(self.heap)

            # Build a small RCL at same distance (actual proximity), sorted by key (penalty-aware)
            layer = [(key, dist, x)]
            while self.heap and self.heap[0][1] == dist and len(layer) < self.rcl_size:
                layer.append(heapq.heappop(self.heap))
            layer.sort(key=lambda t: t[0])
            _, d, px = self.rng.choice(layer)
            # push back unpicked
            for item in layer:
                if item[2] != px:
                    heapq.heappush(self.heap, item)

            # Validate and pick
            if 0 <= px < self.EPlus and px not in self.base and px not in picked:
                picked.append(px)
                # expand neighbors of picked
                self._push_neighbor(px - 1, d + 1)
                self._push_neighbor(px + 1, d + 1)

        # If the heap dries out (e.g., base is dense), do a small rejection-sample fallback
        if len(picked) < k:
            need = k - len(picked)
            picked.extend(self._random_fill(need, exclude=set(self.base).union(picked)))

        # IMPORTANT: do NOT mutate self.base; caller decides to accept or not.
        return picked

    # ---------- internal ----------
    def _seed_initial_neighbors(self):
        for s in self.base:
            self._push_neighbor(s - 1, 1)
            self._push_neighbor(s + 1, 1)

    def _push_neighbor(self, x: int, dist: int):
        if 0 <= x < self.EPlus and x not in self.seen:
            pen = self.add_penalty(x)  # soft bias against recent adds
            key = dist + pen
            heapq.heappush(self.heap, (key, dist, x))
            self.seen.add(x)

    def _random_fill(self, count: int, exclude: Set[int]) -> List[int]:
        out = set()
        attempts = 0
        max_attempts = max(10_000, 5 * count)
        while len(out) < count and attempts < max_attempts:
            x = self.rng.randrange(self.EPlus)
            if x not in exclude:
                out.add(x)
            attempts += 1
        return list(out)


# --------- Helpers: Removal planner with reusable ordering ---------
class RemovalPlanner:
    """
    Precomputes a removal order from the base solution using a chosen metric + IQR,
    then serves prefixes as 'pick_k(k)'. Applies a soft penalty to discourage re-removing
    elements removed very recently.
    """
    __slots__ = ("base", "order", "rng")

    def __init__(self, base: Set[int], metric: str, rcl_size: int, remove_penalty, rng: random.Random):
        self.base = set(base)
        self.rng = rng
        order = self._build_order(metric, remove_penalty)
        # add a tiny RCL jitter in the first block
        head = order[:rcl_size]
        rng.shuffle(head)
        self.order = head + order[rcl_size:]

    def pick_k(self, k: int) -> List[int]:
        return self.order[: min(k, len(self.order))]

    # ---------- internal ----------
    def _build_order(self, metric: str, remove_penalty) -> List[int]:
        arr = sorted(self.base)
        scores: Dict[int, float]
        if metric == "nn_gap":
            scores = _nn_gap_scores(arr)
        elif metric == "value":
            med = arr[len(arr)//2] if len(arr) % 2 else (arr[len(arr)//2 - 1] + arr[len(arr)//2]) / 2
            scores = {x: abs(x - med) for x in arr}
        else:
            raise ValueError("removal_metric must be 'nn_gap' or 'value'")

        xs = list(scores.values())
        low, high = _iqr_bounds(xs)

        def penalized(x: int) -> float:
            return scores[x] - remove_penalty(x)

        outliers = [x for x in arr if scores[x] > high]
        outliers.sort(key=lambda x: penalized(x), reverse=True)
        rest = [x for x in arr if x not in outliers]
        rest.sort(key=lambda x: penalized(x), reverse=True)
        return outliers + rest


# --------- small utilities ---------
def _nn_gap_scores(arr: List[int]) -> Dict[int, float]:
    m = len(arr)
    if m == 1:
        return {arr[0]: float("inf")}
    diffs = [arr[i+1] - arr[i] for i in range(m-1)]
    scores = {arr[0]: diffs[0], arr[-1]: diffs[-1]}
    for i in range(1, m-1):
        scores[arr[i]] = min(diffs[i-1], diffs[i])
    return scores

def _iqr_bounds(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (float("-inf"), float("inf"))
    ys = sorted(xs); n = len(ys)
    def pct(p: float) -> float:
        if n == 1: return ys[0]
        r = (p/100) * (n-1)
        lo = int(r); hi = min(lo+1, n-1); frac = r - lo
        return ys[lo]*(1-frac) + ys[hi]*frac
    q1, q3 = pct(25), pct(75)
    iqr = q3 - q1
    return (q1 - 1.5*iqr, q3 + 1.5*iqr)
