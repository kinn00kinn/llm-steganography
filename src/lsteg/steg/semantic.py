"""Deterministic semantic scaffolding for long-form cover generation.

Small language models often produce fluent local sentences while drifting in
people, place, and chronology over a few hundred tokens. ``SemanticAnchorPlan``
keeps a public, deterministic sequence of semantic goals and periodically
re-prompts the model *outside* the transmitted cover text. Sender and receiver
reconstruct the same hidden prompts from the already-received cover prefix, so
no extra metadata is transmitted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from lsteg.model.interface import LanguageModelBackend

_SENTENCE_ENDINGS = ("。", "！", "？", "!", "?")  # noqa: RUF001


@runtime_checkable
class ChatPromptBackend(LanguageModelBackend, Protocol):
    """Backend capability needed by semantic re-anchoring."""

    def render_chat_prompt(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> str: ...

    def render_chat_continuation(
        self,
        user_prompt: str,
        assistant_prefix: str,
        *,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class SemanticAnchorPlan:
    """A deterministic semantic plan shared by encoder and decoder.

    ``phases`` are not transmitted. After at least ``min_tokens_per_phase``
    visible cover tokens, the next sentence boundary advances to the next
    phase. The receiver observes the same visible prefix and therefore rebuilds
    the same hidden model context at the same cover position.

    ``include_full_outline=False`` is useful with small models: only the current
    goal is shown, leaving lexical freedom and avoiding a long prescriptive
    outline. ``repeat_final_phase=True`` keeps re-anchoring the last open-ended
    carrier phase instead of letting the model drift after the finite outline
    has been exhausted.
    """

    system_prompt: str
    scenario: str
    phases: tuple[str, ...]
    min_tokens_per_phase: int = 40
    include_full_outline: bool = True
    repeat_final_phase: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")
        if not isinstance(self.scenario, str) or not self.scenario.strip():
            raise ValueError("scenario must be a non-empty string")
        if not self.phases or any(
            not isinstance(phase, str) or not phase.strip() for phase in self.phases
        ):
            raise ValueError("phases must contain at least one non-empty string")
        if isinstance(self.min_tokens_per_phase, bool) or not isinstance(
            self.min_tokens_per_phase, int
        ):
            raise TypeError("min_tokens_per_phase must be int")
        if self.min_tokens_per_phase <= 0:
            raise ValueError("min_tokens_per_phase must be positive")
        if not isinstance(self.include_full_outline, bool):
            raise TypeError("include_full_outline must be bool")
        if not isinstance(self.repeat_final_phase, bool):
            raise TypeError("repeat_final_phase must be bool")

    def initial_prompt(self, backend: LanguageModelBackend) -> str:
        """Render the initial hidden chat prompt."""
        chat = _require_chat_backend(backend)
        if self.include_full_outline:
            outline = "\n".join(f"{index + 1}. {phase}" for index, phase in enumerate(self.phases))
            user_prompt = (
                f"{self.scenario}\n\n"
                "次の流れを守り、本文だけを自然につなげて書いてください。\n"
                f"{outline}\n\n"
                f"まず第1段階だけを始めてください：{self.phases[0]}"  # noqa: RUF001
            )
        else:
            user_prompt = (
                f"{self.scenario}\n\n"
                "本文だけを書いてください。今は次の出来事だけを自然に進めてください。"
                "具体的な言い回しや細部は、設定を守る範囲で自由に選んでください。\n"
                f"現在の目標：{self.phases[0]}"  # noqa: RUF001
            )
        return chat.render_chat_prompt(
            user_prompt,
            system_prompt=self.system_prompt,
            enable_thinking=False,
        )

    def continuation_prompt(
        self,
        backend: LanguageModelBackend,
        cover_tokens: Sequence[int],
        *,
        phase_index: int,
    ) -> str:
        """Render a hidden prompt that asks for only the next continuation."""
        if not 0 <= phase_index < len(self.phases):
            raise ValueError("phase_index is outside the semantic plan")
        chat = _require_chat_backend(backend)
        cover_text = backend.detokenize(cover_tokens)
        user_prompt = (
            f"{self.scenario}\n\n"
            "あなたは既に本文を書いています。末尾から直接続きを書き、既存部分の要約・"
            "言い換え・再掲はしないでください。人物・場所・時系列を維持したまま、"
            f"次の目標だけを自然に進めてください：{self.phases[phase_index]}\n"  # noqa: RUF001
            "具体的な表現や細部は、設定を守る範囲で自由に選んでください。"
        )
        return chat.render_chat_continuation(
            user_prompt,
            cover_text,
            system_prompt=self.system_prompt,
            enable_thinking=False,
        )

    def closure_prompt(
        self,
        backend: LanguageModelBackend,
        cover_tokens: Sequence[int],
        *,
        phase_index: int,
    ) -> str:
        """Render a hidden assistant continuation that closes one sentence."""
        if not 0 <= phase_index < len(self.phases):
            raise ValueError("phase_index is outside the semantic plan")
        chat = _require_chat_backend(backend)
        cover_text = backend.detokenize(cover_tokens)
        user_prompt = (
            f"{self.scenario}\n\n"
            "あなたは既に日本語の本文を書いています。末尾の未完の一文だけを自然に完成させ、"
            "できるだけ短く句点「。」まで書いて終了してください。新しい出来事・人物・場所・"
            "英語表現・見出し・教訓は追加せず、既存部分を繰り返さないでください。"
            f"現在の状況：{self.phases[phase_index]}"  # noqa: RUF001
        )
        return chat.render_chat_continuation(
            user_prompt,
            cover_text,
            system_prompt=self.system_prompt,
            enable_thinking=False,
        )

    def new_state(self) -> SemanticAnchorState:
        return SemanticAnchorState(self)

    def current_phase_for_cover(
        self,
        backend: LanguageModelBackend,
        cover_tokens: Sequence[int],
    ) -> int:
        """Replay deterministic boundaries and return the active phase index.

        Replaying phase state must not render chat templates. This method is
        used by tail generation and can otherwise become unnecessarily costly
        for a long cover.
        """
        state = self.new_state()
        for position in range(1, len(cover_tokens) + 1):
            state.observe(backend, cover_tokens[:position], position=position)
        return state.phase_index


@dataclass(slots=True)
class SemanticAnchorState:
    """Mutable per-stream phase state; deterministic from the visible prefix."""

    plan: SemanticAnchorPlan
    phase_index: int = 0
    phase_start_position: int = 0

    def observe(
        self,
        backend: LanguageModelBackend,
        cover_tokens: Sequence[int],
        *,
        position: int,
    ) -> bool:
        """Advance state at a deterministic sentence boundary without rendering."""
        if position - self.phase_start_position < self.plan.min_tokens_per_phase:
            return False
        if not _ends_sentence(backend.detokenize(cover_tokens)):
            return False

        if self.phase_index + 1 < len(self.plan.phases):
            self.phase_index += 1
        elif not self.plan.repeat_final_phase:
            return False

        self.phase_start_position = position
        return True

    def maybe_advance(
        self,
        backend: LanguageModelBackend,
        cover_tokens: Sequence[int],
        *,
        position: int,
    ) -> str | None:
        """Return a replacement hidden prompt when a phase boundary is reached."""
        if not self.observe(backend, cover_tokens, position=position):
            return None
        return self.plan.continuation_prompt(
            backend,
            cover_tokens,
            phase_index=self.phase_index,
        )


def university_lunch_cover_plan(*, min_tokens_per_phase: int = 40) -> SemanticAnchorPlan:
    """Return the original eight-phase Japanese campus diary scaffold."""
    return SemanticAnchorPlan(
        system_prompt=(
            "あなたは自然な現代日本語の短い日記を書く編集者です。事実を簡潔に順番どおり書き、"
            "場所・時刻・登場人物・出来事の因果関係を最後まで一貫させてください。具体的で平易な"
            "語彙を使い、曖昧な指示語、詩的な比喩、恋愛的な描写、身体接触、唐突な教訓、同じ意味の"
            "反復、英語への切り替えは避けてください。構成指示そのものには本文で言及しないでください。"
        ),
        scenario=(
            "語り手は現在の大学生です。時刻は11時45分ごろから13時ごろまでです。同じ大学構内で"
            "起きた一つの日常的な昼食の出来事だけを書きます。登場人物は語り手と同じ研究室の友人"
            "一人だけです。舞台は研究室、研究室から学食までの道、学食、研究室に限定します。"
            "朝・夕方・夜・夕焼け、高校時代への回想、恋愛、事故、知らない人物、劇的な事件は追加しません。"
        ),
        phases=(
            "研究室で課題や作業をしている。時計を見て昼前だと気づく。友人も同じ部屋で作業している。",
            "友人が昼食に行こうと声をかける。短く返事をし、作業を一区切りつける。",
            "二人で研究室を出て学食へ向かう。廊下か構内の様子を一つだけ具体的に書く。",
            "歩きながら授業か研究について短い会話をする。新しい話題や人物は増やさない。",
            "学食に着き、二人がそれぞれ昼食を選んで席に座る。食べ物を一つ具体的に書く。",
            "食事をしながら先ほどの授業か研究の話を少し続ける。大げさな感情描写はしない。",
            "食べ終わって学食を出て、同じ道で研究室へ戻る。昼の時刻のまま進める。",
            "研究室の席に戻って作業を再開する。昼食が小さな気分転換になった程度で自然に締める。",
        ),
        min_tokens_per_phase=min_tokens_per_phase,
    )


def university_lunch_micro_cover_plan(*, min_tokens_per_phase: int = 8) -> SemanticAnchorPlan:
    """Return the existing sentence-scale first-person regression scaffold."""
    return SemanticAnchorPlan(
        system_prompt=(
            "自然な現代日本語で、一人称『私』の日常記録を書いてください。"
            "各段階では指定された事実だけを一文か二文で簡潔に書き、前の文から自然につなげます。"
            "『語り手』『筆者』などの呼び方は使わず、私は『私』と書いてください。"
            "比喩、教訓、恋愛、身体接触、劇的な事件、英語や他言語への切り替え、"
            "同じ内容の言い換え反復は避けてください。"
        ),
        scenario=(
            "平日の11時50分ごろから12時50分ごろまで、大学の研究室から学食へ昼食に行き、"
            "同じ研究室へ戻る出来事だけを書きます。登場人物は私と同じ研究室の友人一人だけです。"
            "天気や時刻を途中で変えず、知らない人物や別の場所を追加しません。"
        ),
        phases=(
            "私は研究室でPCを使って課題の表を直している。時計を見て正午が近いことに気づく。",
            "近くで作業している友人が昼食に行こうと声をかける。私は短く賛成する。",
            "私は作業中のファイルを保存し、PCから手を離して席を立つ。",
            "友人も席を立ち、二人で研究室を出る。",
            "廊下には昼休みに向かう学生の足音が少し増えている、とだけ書く。",
            "二人で階段を下り、学食の方向へ歩く。",
            "歩きながら午前中の課題について一往復だけ短く話す。",
            "学食の入口に着き、掲示されたメニューを見る。",
            "私はカレーを選び、友人は日替わり定食を選ぶ。",
            "二人で空いている席を見つけて座る。",
            "食べ始め、味や温度について日常的な感想を一つだけ書く。",
            "私は午前中に進めていた課題の話を友人に一つ尋ねる。",
            "友人はまだ少し作業が残っていると答える。",
            "私は午後に自分の作業が終わったら確認すると返す。",
            "二人とも昼食を食べ終える。",
            "食器を返却口へ運び、学食を出る。",
            "来たときと同じ道を二人で研究室へ戻る。",
            "研究室に着き、私は自分の席に座る。",
            "私はPCを開き、保存していた課題の続きを再開する。",
            "昼食で少し気分転換できた、という程度の簡潔な一文で自然に終える。",
        ),
        min_tokens_per_phase=min_tokens_per_phase,
    )


def university_lunch_sparse_cover_plan(*, min_tokens_per_phase: int = 8) -> SemanticAnchorPlan:
    """Return a quality-first sparse scaffold for the fixed Qwen3-1.7B model.

    Only semantic state is constrained. Exact wording, food, work details, and
    observations remain model choices so the carrier distribution retains more
    entropy than the prescriptive micro scaffold. The final carrier phase is
    deliberately open-ended and may be re-anchored repeatedly until the payload
    settles; sentence closure is handled only by ``append_sentence_tail``.
    """
    return SemanticAnchorPlan(
        system_prompt=(
            "自然な現代日本語で、大学生本人の一人称『私』の日記を書いてください。"
            "場所・時刻・登場人物の関係を保ち、日常的で具体的な出来事を順番に書いてください。"
            "本文は日本語だけにし、大げさな比喩や教訓、同じ内容の反復は避けてください。"
        ),
        scenario=(
            "平日の正午前から昼過ぎまで、大学の研究室から友人一人と学食へ昼食に行き、"
            "同じ研究室へ戻って作業を続けます。登場人物は私と同じ研究室の友人一人だけです。"
            "場所は研究室、学食までの構内、学食に限定し、時間は昼のまま進めます。"
        ),
        phases=(
            "研究室で作業中に正午が近いことに気づく。作業内容の細部は自然に選ぶ。",
            "友人が昼食に誘い、私は賛成して今の作業をいったん止める。",
            "二人で研究室を出て学食へ向かう。昼休みらしい構内の様子を一つ書く。",
            "歩きながら午前中の授業か研究について短く話す。具体的な話題は自然に選ぶ。",
            "学食でそれぞれ昼食を選んで席に座る。料理や席の細部は自然に選ぶ。",
            "食事をしながら午前中の作業について短いやり取りをする。",
            "食べ終わって学食を出て、来た道を研究室へ戻る。",
            "研究室の席に戻り、止めていた作業を再開する。",
            "昼食前からの作業について小さな確認か修正を一つ進める。",
            "同じ研究室の席で通常の作業を続ける。新しい場所や人物を増やさない。",
            "同じ作業を自然に続け、具体的な小さな操作や確認を一つ書く。まだ結論で締めない。",
        ),
        min_tokens_per_phase=min_tokens_per_phase,
        include_full_outline=False,
        repeat_final_phase=True,
    )


def university_lunch_guarded_sparse_cover_plan(
    *, min_tokens_per_phase: int = 8
) -> SemanticAnchorPlan:
    """Sparse scaffold with a few hard semantic invariants from observed failures.

    The guardrails rule out contradictions that the fixed 1.7B model repeatedly
    produced (eating before reaching the cafeteria, romance/body contact, past
    flashbacks, and time-of-day drift) while leaving wording and incidental
    details unconstrained enough to retain carrier entropy.
    """
    return SemanticAnchorPlan(
        system_prompt=(
            "自然な現代日本語で、大学生本人の一人称『私』の日記を書いてください。"
            "場所・時刻・登場人物の関係を保ち、日常的で具体的な出来事を順番に書いてください。"
            "本文は日本語だけにし、大げさな比喩や教訓、同じ内容の反復は避けてください。"
            "恋愛的な感情や身体接触は書かず、過去の思い出へ脱線しないでください。"
        ),
        scenario=(
            "平日の正午前から昼過ぎまで、大学の研究室から同じ研究室の友人一人と学食へ行き、"
            "昼食後に同じ研究室へ戻って作業を続けます。登場人物は私とその友人だけです。"
            "食事を始めるのは学食に着いてからで、研究室では食べません。"
            "場所は研究室、学食までの構内、学食だけです。朝・夕方・夜へ時刻を移さず、"
            "高校など過去の回想、新しい人物、恋愛、身体接触、劇的な出来事を追加しません。"
        ),
        phases=(
            "研究室で作業中に正午が近いことに気づく。作業内容の細部は自然に選ぶ。",
            "友人が昼食に誘い、私は賛成して今の作業をいったん止める。まだ食事は始めない。",
            "二人で研究室を出て学食へ向かう。昼休みらしい構内の様子を一つ書く。",
            "歩きながら午前中の授業か研究について短く話す。具体的な話題は自然に選ぶ。",
            "学食に着いてから、それぞれ昼食を選んで席に座る。料理や席の細部は自然に選ぶ。",
            "食事をしながら午前中の作業について短いやり取りをする。",
            "食べ終わって学食を出て、来た道を研究室へ戻る。",
            "研究室の席に戻り、止めていた作業を再開する。",
            "昼食前からの作業について小さな確認か修正を一つ進める。",
            "同じ研究室の席で通常の作業を続ける。新しい場所や人物を増やさない。",
            "同じ作業を自然に続け、具体的な小さな操作や確認を一つ書く。まだ結論で締めない。",
        ),
        min_tokens_per_phase=min_tokens_per_phase,
        include_full_outline=False,
        repeat_final_phase=True,
    )


def _require_chat_backend(backend: LanguageModelBackend) -> ChatPromptBackend:
    if not isinstance(backend, ChatPromptBackend):
        raise TypeError("semantic anchoring requires a backend with render_chat_prompt()")
    return backend


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith(_SENTENCE_ENDINGS)
