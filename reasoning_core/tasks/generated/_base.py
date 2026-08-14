from reasoning_core.template import _module_behavior_hash


class GeneratedMixin:
    def behavior_hash(self):
        return _module_behavior_hash(type(self).__module__)


def exact(answer, entry):
    reference = entry["answer"] if isinstance(entry, dict) else entry.answer
    return float(str(answer).strip() == str(reference).strip())
