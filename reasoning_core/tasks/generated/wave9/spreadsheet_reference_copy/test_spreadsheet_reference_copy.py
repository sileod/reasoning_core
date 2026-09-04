from reasoning_core.tasks.generated.wave9.spreadsheet_reference_copy import spreadsheet_reference_copy as m


def test_transformation_relative():
    ref = (False, 1, False, 2)  # B2
    assert m._render_ref(m._shift_ref(ref, 1, 1)) == "C3"


def test_transformation_absolute():
    ref = (True, 0, True, 4)  # $A$4
    assert m._render_ref(m._shift_ref(ref, 2, 3)) == "$A$4"
    assert m._render_ref(m._shift_ref(ref, -1, -1)) == "$A$4"


def test_transformation_mixed_row_absolute():
    ref = (False, 2, True, 5)  # C$5
    assert m._render_ref(m._shift_ref(ref, 1, 2)) == "D$5"


def test_transformation_mixed_col_absolute():
    ref = (True, 0, False, 3)  # $A3
    assert m._render_ref(m._shift_ref(ref, -1, 2)) == "$A5"


def test_num_to_letters_roundtrip():
    assert m._num_to_letters(0) == "A"
    assert m._num_to_letters(25) == "Z"
    assert m._num_to_letters(26) == "AA"
    assert m._num_to_letters(27) == "AB"


def test_generate_and_score():
    task = m.SpreadsheetReferenceCopy()
    for level in (0, 1, 3, 5, 6):
        ex = task.generate_example(level=level)
        assert ex.answer.startswith("=")
        assert m.SpreadsheetReferenceCopy().score_answer(ex.answer, ex) == 1.0
        assert m.SpreadsheetReferenceCopy().score_answer("garbage", ex) < 1.0
        assert ", ".join(ex.answer) != ex.answer
        assert ex.answer != ex.metadata.payload["Original formula"]


def test_answer_space_wide():
    task = m.SpreadsheetReferenceCopy()
    answers = set()
    for _ in range(60):
        answers.add(task.generate_example(level=3).answer)
    assert len(answers) >= 20


def test_config_level_changes():
    c = m.SpreadsheetCopyConfig()
    c0 = c.to_dict()
    c.set_level(3)
    c3 = c.to_dict()
    assert c3 != c0
