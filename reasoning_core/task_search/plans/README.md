# Plans

One `wave*.yaml` per implementation wave: a set of trials, each owning one task directory.
`wave0`–`wave6` were written by hand, before there was a proposer. `wave8` is the first
generated, by `python -m reasoning_core.task_search plan`, from a wave in
`../proposals/archive/`. Nobody should write a new one by hand.

They stay here after they run because `_plan_entries` globs this directory to build the
novelty catalog: 97 of its entries exist only in these files, and deleting a plan tells
the proposer an idea has never been tried. `.legacy/` is outside that glob, so a plan
retires there only once its ideas live on somewhere the catalog still reads.

`WAVE0.md` and `WAVE1.md` are the pre-pipeline idea documents the earliest plans came
from. `WAVE1.md` is still read: `import-legacy` turns it into the `external` proposal wave.
