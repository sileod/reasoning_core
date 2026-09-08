from reasoning_core.tasks.formal_analogies import AnalogicalCaseMatching, _parse_sent


def _metadata(answer_format):
    return {
        "answer_format": answer_format,
        "allow_no_match": False,
        "cases": [
            {
                "id": "M0",
                "context": [("alpha", "b", "e"), ("beta", "a", "e")],
                "consequence": ("alpha", "e", "b"),
            }
        ],
        "query_context": [("delta", "v", "u"), ("epsilon", "u", "z")],
    }


def test_index_prompt_is_compact_and_omits_unused_conclusion():
    prompt = AnalogicalCaseMatching().render_prompt(_metadata("index"))

    assert prompt.startswith("Which case can be embedded into Query?")
    assert "every fact maps to a Query fact" in prompt
    assert "Query may contain additional facts" in prompt
    assert "source link target" not in prompt
    assert "M0: b alpha e, a beta e" in prompt
    assert "e alpha b" not in prompt
    assert "Facts:" not in prompt
    assert "Conclusion:" not in prompt
    assert "Query: v delta u, u epsilon z" in prompt


def test_fact_prompt_keeps_conclusion_as_compact_arrow_target():
    prompt = AnalogicalCaseMatching().render_prompt(_metadata("fact"))

    assert "M0: b alpha e, a beta e -> e alpha b" in prompt
    assert "Query: v delta u, u epsilon z -> ?" in prompt


def test_compact_fact_answer_is_parseable():
    assert _parse_sent("e alpha b") == ("alpha", "e", "b")
