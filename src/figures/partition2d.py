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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass, field

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------------
# Test function: positive almost everywhere, negative inside three blobs.
# The zero level set is the boundary we are trying to localize.
# ----------------------------------------------------------------------------
def f(x, y):
    g = lambda cx, cy, sx, sy: np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
    return 0.55 - (g(-2.5, 3.0, 1.2, 0.9)
                   + g(3.2, 1.5, 1.0, 2.2)
                   + g(-3.5, -3.2, 1.0, 0.8))

BOUNDS = (-5.0, 5.0, -5.0, 5.0)          # xmin, xmax, ymin, ymax
N_PER_REGION = 30                        # samples added to a remaining region per iteration
MIN_SIDE = 0.4                           # don't split below this edge length

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
        new = np.column_stack([rng.uniform(self.xmin, self.xmax, n),
                               rng.uniform(self.ymin, self.ymax, n)])
        self.pts = np.vstack([self.pts, new])
        self.vals = np.concatenate([self.vals, f(new[:, 0], new[:, 1])])

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
    fig.suptitle("Adaptive branch-and-classify partitioning "
                 "(blue = remaining, green = positive, red = negative; "
                 "gray contour = true zero level set)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"wrote {path}")

if __name__ == "__main__":
    plot(run())
