Tasks that pass validation but lose to a constant guess.

`_`-prefixed directories are skipped by `_discover_tasks`, so nothing here reaches
DATASETS. Measured constant-guess reward (most frequent answer, 60 samples/level):

| task | L0 | L3 | L6 | why |
|---|---|---|---|---|
| bipartite_matching | 0.77 | 0.98 | 0.92 | the maximum matching is almost always the whole left side, so the answer is `len(left)` |
| interval_abstract_interpretation | 0.65 | 0.77 | 0.80 | the `exact` class dominates the three-way label |

Fix the generator (deficient graphs; balanced labels), re-measure with
`prior_audit.py`, then move the directory back under `wave0/`.
