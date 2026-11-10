"""Strict input contract for the frozen DeepSeek numerical and task campaign."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from tools.compare_ds4_layer_logits import Record
from tools.freeze_ds4_reference import Digest

CaseId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]
NORMALIZATION = (
    "strip leading/trailing whitespace only; no case folding, fence removal, "
    "answer extraction or numeric coercion"
)


class LogitCase(Record):
    """A held-out prompt and its declared fixed-continuation length."""

    id: CaseId
    prompt_tokens: Annotated[int, Field(ge=1, le=1900)]
    target_positions: Annotated[int, Field(ge=1, le=1024)]
    context_at_last_consumed_target: Annotated[int, Field(ge=2, le=2048)]
    prompt_ids_sha256: Digest

    @model_validator(mode="after")
    def context_matches(self):
        """The declared context must equal the actual prompt plus target count."""
        if (
            self.context_at_last_consumed_target
            != self.prompt_tokens + self.target_positions
        ):
            raise ValueError("declared context differs from prompt plus targets")
        return self


class TaskCase(Record):
    """An exact-answer task with its original prompt hash and normalization."""

    id: CaseId
    prompt_tokens: Annotated[int, Field(ge=1, le=1900)]
    expected: Annotated[str, Field(min_length=1)]
    normalization: Literal[
        "strip leading/trailing whitespace only; no case folding, fence removal, answer extraction or numeric coercion"
    ]
    prompt_ids_sha256: Digest


class Thresholds(Record):
    """Approved thresholds, fixed before collecting fresh qualification data."""

    minimum_positions: Literal[1000]
    minimum_prompts: Literal[5]
    mean_kl_reference_candidate_max_nat: Literal[0.01]
    top1_agreement_min: Literal[0.99]
    task_gate: Literal["retain every baseline pass across the eight fixed tasks"]


class TemplateSettings(Record):
    """The prepared corpus uses the pinned template's non-thinking mode."""

    thinking: Literal[False]
    enable_thinking: Literal[False]


class QualityInputs(Record):
    """Corpus inputs only; a separate runtime campaign must seal their hash."""

    status: str
    logit_cases: Annotated[list[LogitCase], Field(min_length=5)]
    task_cases: Annotated[list[TaskCase], Field(min_length=8, max_length=8)]
    thresholds_from_approved_plan: Thresholds
    tokenizer_assets_sha256: dict[str, Digest]
    template_settings: TemplateSettings
    source_code_sha256: dict[str, Digest]
    limits: str

    @model_validator(mode="after")
    def distinct_cases(self):
        """Reject duplicated cases and fewer positions than the approved corpus."""
        for cases in (self.logit_cases, self.task_cases):
            if len({case.id for case in cases}) != len(cases):
                raise ValueError("duplicate quality case identity")
        if sum(case.target_positions for case in self.logit_cases) < 1000:
            raise ValueError("quality corpus has fewer than 1000 fixed positions")
        if set(self.tokenizer_assets_sha256) != {
            "vocab.json",
            "merges.txt",
            "tokenizer.json",
            "token_type.json",
            "chat_template.jinja",
        }:
            raise ValueError("quality corpus tokenizer asset identities are incomplete")
        if set(self.source_code_sha256) != {
            "tools/strata_tokenizer.py",
            "serve/frontend.py",
        }:
            raise ValueError(
                "quality corpus preparation source identities are incomplete"
            )
        if {case.id for case in self.logit_cases} & {
            "task-" + case.id for case in self.task_cases
        }:
            raise ValueError("logit and task output identities overlap")
        return self
