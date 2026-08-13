from reasoning_core.template import _module_behavior_hash


class GeneratedMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def behavior_hash(self):
        module = next(cls.__module__ for cls in type(self).__mro__ if cls.__name__.endswith("Mixin"))
        return _module_behavior_hash(module)


def exact(answer, entry):
    reference = entry["answer"] if isinstance(entry, dict) else entry.answer
    return float(str(answer).strip() == str(reference).strip())
