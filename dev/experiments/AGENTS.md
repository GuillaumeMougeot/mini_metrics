# Frozen experiments

The existing `threshold_stability/` and `thresholds/` studies are frozen reference
snapshots. Preserve their policies, seeds, defaults, test expectations and source
as recorded in `FREEZE.md`. Do not routinely refactor, extend or repair them while
working on the package. This is a maintenance boundary, not a file permission lock.

Run them for replay when useful, writing results to a fresh local directory.
Record package/source versions: freezing scripts does not freeze their imports.
For a new hypothesis, create a separately named study with its own protocol and
cite the frozen comparator. Make an intentional compatibility repair only when
needed for an authorized replay, recording the change and new hashes separately.

Continue supported correctness and performance work in `tests/` and `benchmarks/`.
Do not recreate the retired experiments listed in `README.md` as incidental cleanup.
