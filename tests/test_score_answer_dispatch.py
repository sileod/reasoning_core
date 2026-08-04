import json

import pytest

from reasoning_core import score_answer
from reasoning_core.template import edict


class NoDeepcopy:
    def __deepcopy__(self, memo):
        raise NotImplementedError("proxy cannot be deep-copied")


def test_dispatch_decodes_string_metadata_without_deepcopying_the_row():
    entry = edict(
        answer="crate",
        metadata=json.dumps({"_task": "belief_tracking"}),
        unrelated_proxy=NoDeepcopy(),
    )

    assert score_answer("crate", entry) == 1.0
    assert isinstance(entry.metadata, str)


def test_regression_proxy_really_rejects_deepcopy():
    import copy

    with pytest.raises(NotImplementedError):
        copy.deepcopy(NoDeepcopy())
