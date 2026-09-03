import random
from pathlib import Path

from reasoning_core.tasks.generated.wave8.posix_acl_permission.posix_acl_permission import (
    PosixAclPermission,
    PosixAclPermissionV2Config,
)

random.seed(737652980)
task = PosixAclPermission(config=PosixAclPermissionV2Config(seed=0))

out = Path(__file__).with_name("samples_P048v2.md")
with out.open("w") as fh:
    for level in (0, 2, 5):
        fh.write(f"## Level {level}\n\n")
        for _ in range(2):
            ex = task.generate_example(level=level)
            fh.write("### Prompt\n\n")
            fh.write(ex.prompt + "\n\n")
            fh.write("### Answer\n\n")
            fh.write(ex.answer + "\n\n")
