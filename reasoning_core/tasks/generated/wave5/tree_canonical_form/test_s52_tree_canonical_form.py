from reasoning_core.tasks.generated.wave5.s52_tree_canonical_form.s52_tree_canonical_form import (
    TreeCanonicalForm, TreeCanonicalFormConfig, canonical, build_tree, relabeled_parents
)
import random


def test_relabel_preserves_shape():
    random.seed(7)
    for _ in range(20):
        n = 9
        parents = [random.randrange(i) for i in range(1, n)]
        p2 = relabeled_parents(parents, random)
        assert canonical(build_tree(p2), 0) == canonical(build_tree(parents), 0)


def test_gold_scores():
    t = TreeCanonicalForm()
    for _ in range(20):
        x = t.generate_example()
        assert t.score_answer(x.answer, x) == 1.0


def test_junk_scores_zero():
    t = TreeCanonicalForm()
    x = t.generate_example()
    assert t.score_answer("", x) < 1.0
    assert t.score_answer("garbage", x) < 1.0


def test_leaves_and_internal():
    assert canonical(build_tree([0, 1, 2]), 0) == "(((())))"
    assert canonical(build_tree([0, 0, 0]), 0) == "(()()())"


def test_difficulty_changes():
    import random
    random.seed(1)
    c = TreeCanonicalFormConfig()
    c.set_level(0)
    n0 = int(c.n_nodes)
    c.set_level(5)
    n5 = int(c.n_nodes)
    assert n5 > n0
