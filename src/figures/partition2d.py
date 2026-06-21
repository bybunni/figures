"""
Adaptive 2D space partitioning, in the spirit of Part-X style branch-and-classify
algorithms. Not a faithful reproduction -- a minimal, readable illustration of the
general idea:

  1. Start with the whole design space as a single "remaining" region.
  2. Each iteration, sample inside every remaining region and try to CLASSIFY it:
       - all evidence says f > 0  -> positive region (safe, green)
       - all evidence says f < 0  -> negative region (failure, red)
       - mixed / uncertain        -> BRANCH it into children and recurse next iteration
  3. Classified regions can be RE-OPENED if later samples contradict the label.

The classifier here is deliberately simple (sample min/max plus a margin that
shrinks with sample count); Part-X proper uses Gaussian-process estimates of the
level set with statistical guarantees, but the control flow is the same.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------------
# Test function: positive almost everywhere, negative inside three ellipsoids.
# The zero robustness limit is the boundary we are trying to localize.
# ----------------------------------------------------------------------------
ELLIPSOIDS = (
    (-2.5, 3.0, 1.2, 0.9),
    (3.2, 1.5, 1.0, 2.2),
    (-3.5, -3.2, 1.0, 0.8),
)

def f(x, y):
    margins = [
        ((np.asarray(x) - cx) / sx) ** 2 + ((np.asarray(y) - cy) / sy) ** 2 - 1
        for cx, cy, sx, sy in ELLIPSOIDS
    ]
    robustness = np.minimum.reduce(margins)
    if np.asarray(robustness).ndim == 0:
        return float(robustness)
    return robustness

def evaluate(x, y):
    robustness = f(x, y)
    robustness_array = np.asarray(robustness)
    classification = np.where(robustness_array < 0, "negative", "positive")
    distance_to_limit = np.abs(robustness_array)

    if robustness_array.ndim == 0:
        return str(classification.item()), float(robustness), float(distance_to_limit)
    return classification, robustness, distance_to_limit

BOUNDS = (-5.0, 5.0, -5.0, 5.0)          # xmin, xmax, ymin, ymax
N_PER_REGION = 30                        # samples added to a remaining region per iteration
MIN_SIDE = 0.4                           # don't split below this edge length
CANDIDATES_PER_SAMPLE = 20               # candidate pool for robustness-biased sampling
MIN_CANDIDATES = 100
ROBUSTNESS_WEIGHT_EPS = 1e-6

def _robustness_sample_weights(robustness):
    distance_to_limit = np.abs(np.asarray(robustness, dtype=float))
    weights = 1.0 / (distance_to_limit + ROBUSTNESS_WEIGHT_EPS)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(distance_to_limit.shape, 1.0 / distance_to_limit.size)
    return weights / total

def _candidate_count(n):
    return max(n, MIN_CANDIDATES, n * CANDIDATES_PER_SAMPLE)

# ----------------------------------------------------------------------------
# Region bookkeeping
# ----------------------------------------------------------------------------
@dataclass
class Region:
    xmin: float; xmax: float; ymin: float; ymax: float
    status: str = "remaining"            # remaining | positive | negative
    pts: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    vals: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def sample(self, n):
        if n <= 0:
            return

        n_candidates = _candidate_count(n)
        candidates = np.column_stack([rng.uniform(self.xmin, self.xmax, n_candidates),
                                      rng.uniform(self.ymin, self.ymax, n_candidates)])
        candidate_vals = f(candidates[:, 0], candidates[:, 1])
        weights = _robustness_sample_weights(candidate_vals)
        picked = rng.choice(n_candidates, size=n, replace=False, p=weights)
        new = candidates[picked]
        self.pts = np.vstack([self.pts, new])
        self.vals = np.concatenate([self.vals, candidate_vals[picked]])

    def classify(self):
        """Label the region if all evidence agrees, with a margin that shrinks
        as evidence accumulates (a crude stand-in for a confidence bound)."""
        if len(self.vals) < 10:
            return "remaining"
        margin = 0.35 / np.sqrt(len(self.vals))
        if self.vals.min() > margin:
            return "positive"
        if self.vals.max() < -margin:
            return "negative"
        return "remaining"

    def split(self):
        """Branch along the longest axis; children inherit contained samples."""
        if max(self.xmax - self.xmin, self.ymax - self.ymin) < MIN_SIDE:
            return [self]                # too small to split, stays remaining
        if (self.xmax - self.xmin) >= (self.ymax - self.ymin):
            mid = 0.5 * (self.xmin + self.xmax)
            kids = [Region(self.xmin, mid, self.ymin, self.ymax),
                    Region(mid, self.xmax, self.ymin, self.ymax)]
        else:
            mid = 0.5 * (self.ymin + self.ymax)
            kids = [Region(self.xmin, self.xmax, self.ymin, mid),
                    Region(self.xmin, self.xmax, mid, self.ymax)]
        for k in kids:
            mask = ((self.pts[:, 0] >= k.xmin) & (self.pts[:, 0] <= k.xmax) &
                    (self.pts[:, 1] >= k.ymin) & (self.pts[:, 1] <= k.ymax))
            k.pts, k.vals = self.pts[mask], self.vals[mask]
        return kids

# ----------------------------------------------------------------------------
# Main loop: branch and classify
# ----------------------------------------------------------------------------
def run(n_iters=9, snapshot_at=(1, 3, 6, 9)):
    regions = [Region(*BOUNDS)]
    snapshots = {}

    for k in range(1, n_iters + 1):
        nxt = []
        for r in regions:
            if r.status != "remaining":
                # light re-audit of classified regions: a few cheap samples,
                # re-open the region if the label no longer holds
                r.sample(3)
                if r.classify() not in (r.status, "remaining") or \
                   (r.classify() == "remaining" and len(r.vals) > 40 and
                    np.sign(r.vals.min()) != np.sign(r.vals.max())):
                    r.status = "remaining"
                nxt.append(r)
                continue

            r.sample(N_PER_REGION)
            label = r.classify()
            if label != "remaining":
                r.status = label
                nxt.append(r)
            else:
                nxt.extend(r.split())
        regions = nxt
        if k in snapshot_at:
            snapshots[k] = [Region(r.xmin, r.xmax, r.ymin, r.ymax,
                                   r.status, r.pts.copy(), r.vals.copy())
                            for r in regions]
        counts = {s: sum(r.status == s for r in regions)
                  for s in ("remaining", "positive", "negative")}
        print(f"iter {k}: {len(regions):4d} regions  {counts}")
    return snapshots

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
COLORS = {"remaining": "#b9c2f0", "positive": "#cdeccd", "negative": "#f3b9b9"}

def plot(snapshots, path="partition_iterations.png"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(snapshots), figsize=(4.6 * len(snapshots), 4.6))
    xs = np.linspace(BOUNDS[0], BOUNDS[1], 400)
    ys = np.linspace(BOUNDS[2], BOUNDS[3], 400)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)

    for ax, (k, regs) in zip(np.atleast_1d(axes), sorted(snapshots.items())):
        for r in regs:
            ax.add_patch(Rectangle((r.xmin, r.ymin), r.xmax - r.xmin,
                                   r.ymax - r.ymin, facecolor=COLORS[r.status],
                                   edgecolor="black", linewidth=0.5))
            if len(r.pts):
                ax.scatter(r.pts[:, 0], r.pts[:, 1], s=1.5,
                           c=np.where(r.vals < 0, "crimson", "navy"), zorder=3)
        ax.contour(X, Y, Z, levels=[0.0], colors="dimgray", linewidths=1.2, zorder=4)
        ax.set_xlim(BOUNDS[0], BOUNDS[1]); ax.set_ylim(BOUNDS[2], BOUNDS[3])
        ax.set_aspect("equal")
        ax.set_title(f"iteration k = {k}")
    fig.suptitle("Adaptive robustness-aware branch-and-classify partitioning "
                 "(blue = remaining, green = positive, red = negative; "
                 "gray contour = zero robustness limit)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

def _project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()

def default_output_path():
    return _project_root() / "output" / "partition_iterations.png"

def _positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed

def _default_snapshot_at(n_iters):
    return tuple(sorted({k for k in (1, 3, 6, n_iters) if k <= n_iters}))

def _parse_snapshot_at(value, n_iters):
    if value is None:
        return _default_snapshot_at(n_iters)

    snapshots = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            raise argparse.ArgumentTypeError("snapshots must not contain empty values")
        try:
            snapshot = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("snapshots must be integers") from exc
        if snapshot <= 0:
            raise argparse.ArgumentTypeError("snapshots must be positive")
        if snapshot > n_iters:
            raise argparse.ArgumentTypeError("snapshots must be within iterations")
        snapshots.append(snapshot)

    if not snapshots:
        raise argparse.ArgumentTypeError("snapshots must not be empty")
    return tuple(sorted(set(snapshots)))

def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run the adaptive 2D partitioning demo and save a figure.",
    )
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=9,
        help="number of branch-and-classify iterations to run (default: 9)",
    )
    parser.add_argument(
        "--snapshots",
        help="comma-separated iterations to include in the output figure",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help=f"path for the generated figure (default: {default_output_path()})",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        snapshot_at = _parse_snapshot_at(args.snapshots, args.iterations)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    snapshots = run(n_iters=args.iterations, snapshot_at=snapshot_at)
    plot(snapshots, path=args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
