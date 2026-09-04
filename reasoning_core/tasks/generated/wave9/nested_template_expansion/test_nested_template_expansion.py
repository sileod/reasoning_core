import os
import random

from reasoning_core.tasks.generated.wave9.nested_template_expansion.nested_template_expansion import (
    NestedTemplateExpansion,
    expand_template,
)


def test_expand_basic_substitution():
    assert expand_template("{{ a }}", {"a": "hello"}) == "hello"


def test_expand_nested():
    # v1 references v0
    assert expand_template("{{ v1 }}", {"v0": "x", "v1": "{{ v0 }}y"}) == "xy"


def test_expand_default_used_and_not():
    vars_ = {"a": "val"}
    assert expand_template("{{ a | def }}", vars_) == "val"
    assert expand_template("{{ b | def }}", vars_) == "def"


def test_expand_conditional_truthy():
    vars_ = {"c": "5", "c0": ""}
    assert expand_template("{{ c ? yes | no }}", vars_) == "yes"
    assert expand_template("{{ c0 ? yes | no }}", vars_) == "no"


def test_expand_conditional_false_word():
    assert expand_template("{{ f ? yes | no }}", {"f": "false"}) == "no"


def test_escape_braces():
    # escaped braces stay literal; backslash removed
    assert expand_template("\\{{ literal \\}}", {}) == "{ literal }"


def test_generate_entry_and_score():
    t = NestedTemplateExpansion()
    ex = t.generate_example()
    assert t.score_answer(ex.answer, ex) == 1.0


def test_gold_is_fixed_point():
    t = NestedTemplateExpansion()
    for _ in range(20):
        ex = t.generate_example()
        again = expand_template(ex.answer, ex.metadata.variables)
        assert again == ex.answer


def test_wrong_answer_scores_zero():
    t = NestedTemplateExpansion()
    ex = t.generate_example()
    assert t.score_answer("garbage", ex) == 0.0
    assert t.score_answer("", ex) == 0.0


def test_deterministic_under_seed():
    random.seed(12345)
    t = NestedTemplateExpansion()
    t.config.set_level(3)
    a = t.generate_example().answer
    random.seed(12345)
    t2 = NestedTemplateExpansion()
    t2.config.set_level(3)
    b = t2.generate_example().answer
    assert a == b


def test_nested_inner_default():
    # outer reference has a default that itself holds a reference
    assert expand_template("{{ a | {{ b }} }}", {"b": "bee"}) == "bee"
    assert expand_template("{{ a | {{ b }} }}", {"a": "aye", "b": "bee"}) == "aye"


def test_conditional_with_nested_condval():
    # cond value references another variable and resolves before truthiness
    assert expand_template("{{ c ? X | Y }}", {"c": "0", "z": "5"}) == "Y"
    assert expand_template("{{ c ? X | Y }}", {"c": "{{z}}", "z": "3"}) == "X"


def test_multiple_operators_and_separators():
    template = "{{a}}/{{b | D}} - {{c ? Y | N}}"
    assert expand_template(template, {"a": "1", "b": "2", "c": "9"}) == "1/2 - Y"
    assert expand_template(template, {"a": "1", "b": "2", "c": "0"}) == "1/2 - N"
    assert expand_template(template, {"a": "1", "c": "9"}) == "1/D - Y"


def test_no_dangling_braces():
    t = NestedTemplateExpansion()
    for _ in range(30):
        ex = t.generate_example()
        assert "}}" not in ex.answer.replace("{literal", "")


def test_every_level_generates():
    t = NestedTemplateExpansion()
    for level in (0, 1, 2, 3, 4, 5, 6):
        t.config.set_level(level)
        ex = t.generate_example()
        assert t.score_answer(ex.answer, ex) == 1.0


def test_score_rejects_fence_and_garbage():
    t = NestedTemplateExpansion()
    ex = t.generate_example()
    assert t.score_answer("```" + ex.answer + "```", ex) == 1.0
    assert t.score_answer("```python\n" + ex.answer + "\n```", ex) == 1.0
    assert t.score_answer("not it", ex) == 0.0
