"""Deterministic scenario bank for Japanese prose distillation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Scenario:
    """One teacher prompt with a scenario-level split to prevent leakage."""

    scenario_id: str
    category: str
    split: str
    prompt: str


_CATEGORIES = (
    "大学での作業",
    "図書館での用事",
    "買い物",
    "家での家事",
    "近所の散歩",
    "電車での移動",
    "カフェでの休憩",
    "友人との短い会話",
    "天気にまつわる日常",
    "小さな失敗と立て直し",
)

_SETTINGS = (
    "平日の午前中",
    "正午前後",
    "平日の午後",
    "夕方になる少し前",
    "休日の午前中",
    "休日の午後",
)

_EVENTS = (
    "予定していた作業を一区切りつける",
    "必要な物を一つ買いに行く",
    "短い休憩を取ってから作業に戻る",
    "友人と数分だけ近況を話す",
    "忘れ物に気づいて落ち着いて対処する",
    "混雑を避けて少し経路を変える",
    "飲み物や軽い食事を選ぶ",
    "資料や持ち物を整理する",
    "天気の変化に合わせて予定を少し変える",
    "終わっていなかった小さな用事を片づける",
)

_TONES = (
    "特別な事件ではない、落ち着いた日常として",
    "具体的な動作と周囲の様子を中心に",
    "大げさな感情表現を避けて簡潔に",
    "一つの出来事だけに焦点を絞って",
    "時間の順序が分かるように",
)


def build_scenarios(count: int, *, seed: int) -> list[Scenario]:
    """Build unique prompts deterministically without asking an LLM to invent prompts."""
    if not 1 <= count <= len(_CATEGORIES) * len(_SETTINGS) * len(_EVENTS) * len(_TONES):
        raise ValueError("scenario count exceeds the deterministic scenario space")

    combinations = [
        (category, setting, event, tone)
        for category in _CATEGORIES
        for setting in _SETTINGS
        for event in _EVENTS
        for tone in _TONES
    ]
    rng = random.Random(seed)
    rng.shuffle(combinations)

    scenarios: list[Scenario] = []
    for category, setting, event, tone in combinations[:count]:
        identity = f"{category}|{setting}|{event}|{tone}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        scenario_id = f"jp-prose-{digest[:12]}"
        split = "eval" if int(digest[-2:], 16) < 26 else "train"
        prompt = (
            f"状況: {category}。時刻: {setting}。中心となる出来事: {event}。\n"
            f"{tone}、自然な現代日本語の一人称文章を書いてください。"
            "本文は250〜500文字程度にし、本文だけを出力してください。"
            "登場人物・場所・時刻を途中で変えず、知らない人物や劇的な事件を追加しないでください。"
            "詩的な教訓、恋愛への脱線、不自然な身体接触、文章作成についての説明は避けてください。"
        )
        scenarios.append(Scenario(scenario_id, category, split, prompt))
    return scenarios
