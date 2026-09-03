from reasoning_core.tasks.generated.wave8.red_black_black_height.red_black_black_height import (
    RedBlackBlackHeight, RedBlackBlackHeightConfig, valid_rb, black_height, RBNode,
)


def _parse(s):
    pos = [0]

    def parse_node():
        seg = s
        i = pos[0]
        if seg[i:i + 1] == '(':
            pos[0] = i + 2
            return None
        assert seg[i] == '['
        i += 1
        j = i
        while seg[j].isdigit():
            j += 1
        key = int(seg[i:j])
        color = seg[j]
        j += 1
        assert seg[j] == ','
        pos[0] = j + 1
        left = parse_node()
        assert seg[pos[0]] == ','
        pos[0] += 1
        right = parse_node()
        assert seg[pos[0]] == ']'
        pos[0] += 1
        n = RBNode(key)
        n.color = 0 if color == 'B' else 1
        n.left, n.right = left, right
        if left:
            left.parent = n
        if right:
            right.parent = n
        return n

    return parse_node()


def test_gold_scores_one():
    t = RedBlackBlackHeight()
    x = t.generate_example()
    assert t.score_answer(x.answer, x) == 1.0


def test_junk_scores_zero():
    t = RedBlackBlackHeight()
    x = t.generate_example()
    assert t.score_answer('', x) == 0.0
    assert t.score_answer('garbage', x) == 0.0
    assert t.score_answer(None, x) == 0.0


def test_serialized_is_valid_rb_and_height_matches():
    t = RedBlackBlackHeight()
    for _ in range(25):
        x = t.generate_example()
        root = _parse(x.metadata.payload['serialized'])
        assert root is not None
        ok, _ = valid_rb(root)
        assert ok

        nodes = []

        def collect(n):
            if n is None:
                return
            nodes.append(n)
            collect(n.left)
            collect(n.right)

        collect(root)
        target = next(n for n in nodes if n.key == x.metadata.payload['node_key'])
        assert black_height(target) == int(x.answer)


def test_difficulty_changes():
    c = RedBlackBlackHeightConfig()
    c.set_level(0)
    n0 = c.n_nodes
    c.set_level(6)
    assert c.n_nodes > n0
