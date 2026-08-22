# Contributing

Contributions that improve correctness, portability, tests or documentation are
welcome.

1. Create a focused branch.
2. Install `.[dev]` and run `pytest` and `ruff check .`.
3. Add a regression test for behavioral changes.
4. Keep datasets, checkpoints, identifiers and absolute acquisition paths out
   of commits.
5. Describe scientific assumptions and distinguish validation metrics from
   test metrics in the pull request.

Changes to the Stokes convention, Cloude transform, incident-bank orientation
or ridge reconstruction require an explicit mathematical note and round-trip
tests. Do not silently change a convention to improve a metric.
