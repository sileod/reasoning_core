import ast

import reasoning_core.tasks.generated.wave9.alias_mutation_tracking.alias_mutation_tracking as mod


def _simulate(name, entry):
    cfg = mod.AliasMutationConfig()
    task = mod.AliasMutation()
    task.config = cfg
    e = task.generate_example()
    return e


def test_gold_scores_one():
    task = mod.AliasMutation()
    for _ in range(40):
        e = task.generate_example()
        assert task.score_answer(e.answer, e) == 1.0, e.answer


def test_answer_is_list_of_ints():
    task = mod.AliasMutation()
    for _ in range(20):
        e = task.generate_example()
        parsed = ast.literal_eval(e.answer)
        assert isinstance(parsed, list)
        assert all(isinstance(x, int) and not isinstance(x, bool) for x in parsed)


def test_query_var_present_in_stated_variables():
    task = mod.AliasMutation()
    for _ in range(20):
        e = task.generate_example()
        assert e.metadata.query in e.metadata.variables


def test_wrong_answer_scores_zero():
    task = mod.AliasMutation()
    e = task.generate_example()
    assert task.score_answer("nonsense", e) == 0.0
    assert task.score_answer("", e) == 0.0


def test_simulation_matches_spec():
    task = mod.AliasMutation()
    for _ in range(30):
        e = task.generate_example()
        v = {k: list(val) for k, val in e.metadata.variables.items()}
        for op in e.metadata.operations:
            if ".append(" in op:
                var = op.split(".append")[0]
                v[var].append(int(op.split("(", 1)[1][:-1]))
            elif "[" in op and op.index("[") < op.index(" = "):
                var = op[:op.index("[")]
                idx = int(op[op.index("[") + 1:op.index("]")])
                val = int(op.split(" = ")[1])
                v[var][idx] = val
            elif " = " in op:
                dst, rest = op.split(" = ")
                if rest.strip().startswith("["):
                    v[dst] = ast.literal_eval(rest)
                else:
                    v[dst] = v[rest.strip()]
        assert v[e.metadata.query] == ast.literal_eval(e.answer)
