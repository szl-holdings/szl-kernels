from pathlib import Path


def test_eval_status_messages_are_ascii_console_safe() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "eval.py").read_text(
        encoding="utf-8"
    )
    assert "\u2265" not in source
    assert "\u2264" not in source
    assert "OK >=0.90" in source
    assert "OK <=0.02" in source


def test_eval_replay_copies_the_input_contract() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "eval.py").read_text(
        encoding="utf-8"
    )
    assert 'scripts/corpus_inputs.py", f"{td}/scripts/corpus_inputs.py"' in source
    assert 'input_contract = "legacy-v1-replay"' in source
    assert 'input_contract = "canonical-v2"' in source
