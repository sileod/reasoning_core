# Battery manifests

A manifest is ordered data: a list of legs, each a frozen `.jsonl` file plus the kind of
scoring applied to it. The runner lives in `reasoning_core/evaluation/battery.py`; nothing
here is code.

`copyfree_battery_v8_tiny.json` is what current results are measured on, through
`default_battery()`. `paper_battery.json` is the 21-leg suite of the first influence paper,
reachable through `paper_battery()`. Everything else is archived: kept so an old
measurement can be rebuilt, referenced by nothing.

## Why the filenames still carry version numbers

Each manifest pins its own `name`, and `EvalBattery.identifier` hashes that name together
with the leg identities and `max_length`. The identifier is recorded in every measured
arm's `eval_ids`, so `copyfree_battery_v8_tiny` is part of an identity that already exists
in shipped results. Renaming the file to something semantic would either orphan those
results or leave a filename that disagrees with the identity inside it.

Since the name is pinned in the JSON rather than derived from the filename, the identifier
does not depend on where the file lives — which is what made moving this directory safe.
Adding a manifest is therefore free; renaming or editing an existing one is not.

Note that `copyfree_battery_v8.json`, `v9`, and `v10` all declare the name
`copyfree_battery_v8`, so three distinct batteries report identifiers under one name and
are told apart only by the digest. Frozen for the reason above; do not repeat the pattern.

## Adding one

Give it a `name` no existing manifest uses, put its legs in `battery_legs.zip` beside this
directory, and address it through `load_battery_manifest()`. Set `max_length` deliberately:
it is part of the identifier, and results built at different values do not pool.
