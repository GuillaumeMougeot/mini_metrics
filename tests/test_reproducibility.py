"""Results must not depend on Python's per-process string-hash randomization."""

import os
import subprocess
import sys

SCRIPT = """
import numpy as np
from mini_metrics.data import MetricDF
from mini_metrics.metrics import evaluate_file
rng = np.random.default_rng(0)
n = 2000
labels = rng.choice([f"sp{i}" for i in range(200)], n)
preds = np.where(rng.uniform(size=n) < 0.5, labels, rng.choice([f"sp{i}" for i in range(220)], n))
df = MetricDF({"instance_id": np.arange(n), "filename": np.array([f"f{i}" for i in range(n)], dtype=object),
    "level": np.zeros(n, dtype=np.int64), "label": labels.astype(object), "prediction": preds.astype(object),
    "confidence": rng.uniform(size=n), "threshold": np.zeros(n)})
for per_class in (False, True):
    m = evaluate_file(df, threshold=[0.4], per_class=per_class, simple=True, hierarchical=False, verbose=0, pattern="^(f1|precision)$")
    print(repr(m))
"""


def run(seed):
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    return subprocess.run(
        [sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, check=True
    ).stdout


def test_metrics_and_class_order_are_independent_of_hash_seed():
    reference = run(0)
    assert all(run(seed) == reference for seed in (1, 2, 3))
