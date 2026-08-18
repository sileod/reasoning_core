#!/usr/bin/env bash
# One-command minimal install for writing/validating a task generator.
#
#   scripts/install_authoring.sh          # in a clone
#   scripts/install_authoring.sh --pypi   # no clone: pull the released wheel instead
#
# Minimal is deliberately NOT the default install. `pip install reasoning-core` stays
# batteries-included, because the package is a collection of heterogeneous generators and a user
# instantiating an arbitrary task should not hit a missing dependency. A `[minimal]` extra cannot
# express this: extras are ADDITIVE, so reasoning-core[minimal] would install the full stack plus
# the small list. --no-deps is the only mechanism that actually installs less.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ="$HERE/requirements/task-authoring.txt"

if [ "${1:-}" = "--pypi" ]; then
  pip install reasoning-core --no-deps
else
  pip install -e "$HERE" --no-deps
fi
# read the list from the file when present so it never drifts from the repo
if [ -f "$REQ" ]; then
  pip install -r "$REQ"
else
  pip install easydict inflection wrapt xxhash appdirs
fi

python - <<'PY'
from reasoning_core.template import Task
print("authoring install ok: reasoning_core.template imports without the full stack")
PY
cat <<'MSG'

Now validate a generator:

  python -c "from reasoning_core.tasks.generated.pattern_induction import PatternInduction as T; T().validate(); print('ok')"

MSG
