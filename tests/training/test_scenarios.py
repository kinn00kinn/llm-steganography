from __future__ import annotations

from lsteg.training.scenarios import build_scenarios


def test_scenario_bank_is_deterministic_and_unique() -> None:
    first = build_scenarios(600, seed=20260816)
    second = build_scenarios(600, seed=20260816)
    assert first == second
    assert len({item.scenario_id for item in first}) == 600
    assert {item.split for item in first} == {"train", "eval"}


def test_scenario_split_is_scenario_level() -> None:
    scenarios = build_scenarios(120, seed=11)
    by_id = {item.scenario_id: item.split for item in scenarios}
    assert len(by_id) == len(scenarios)


def test_scenario_prompt_requests_plain_japanese_prose() -> None:
    scenario = build_scenarios(1, seed=7)[0]
    assert "自然な現代日本語" in scenario.prompt
    assert "本文だけ" in scenario.prompt
    assert "250〜500文字" in scenario.prompt
