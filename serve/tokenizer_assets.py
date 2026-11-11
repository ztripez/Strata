"""Load pinned DeepSeek exported assets through Strata's canonical tokenizer."""

import hashlib
import sys
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from tools.strata_tokenizer import Tokenizer

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ASSETS = {
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "token_type.json",
    "chat_template.jinja",
}


class Record(BaseModel):
    """Strict immutable boundary used by the experimental serving adapter."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class TokenizerPins(Record):
    """Expected exported asset bytes and the authoritative tokenizer source."""

    assets_sha256: dict[str, Digest]
    source_sha256: Digest

    @model_validator(mode="after")
    def complete_assets(self):
        """An omitted or extra asset cannot become a partially verified tokenizer."""
        if set(self.assets_sha256) != ASSETS:
            raise ValueError("DeepSeek tokenizer requires exactly five pinned assets")
        return self


class TokenizerConfiguration(Record):
    """Exported DeepSeek configuration, without inferred special-token defaults."""

    pre: Literal["deepseek3"]
    special_ids: dict[str, int]


def load_deepseek(assets: Path, pins: TokenizerPins) -> Tokenizer:
    """Hash and parse the same buffers; never reload a different file view."""
    buffers = {
        path.name: path.read_bytes() for path in assets.iterdir() if path.is_file()
    }
    actual = {name: hashlib.sha256(raw).hexdigest() for name, raw in buffers.items()}
    if actual != pins.assets_sha256:
        raise ValueError("tokenizer asset set or bytes differ from pinned assets")
    source = Path(sys.modules[Tokenizer.__module__].__file__)
    if hashlib.sha256(source.read_bytes()).hexdigest() != pins.source_sha256:
        raise ValueError("authoritative tokenizer source differs from pinned source")
    config = TokenizerConfiguration.model_validate_json(buffers["tokenizer.json"])
    if config.special_ids.get("eos") != 1:
        raise ValueError("DeepSeek EOS identity differs")
    vocabulary = TypeAdapter(dict[str, int]).validate_json(
        buffers["vocab.json"], strict=True
    )
    if sorted(vocabulary.values()) != list(range(129280)):
        raise ValueError("tokenizer is not the pinned contiguous DeepSeek vocabulary")
    types = TypeAdapter(list[int]).validate_json(
        buffers["token_type.json"], strict=True
    )
    if len(types) != len(vocabulary):
        raise ValueError("tokenizer type count differs from vocabulary")
    tokens = [""] * len(vocabulary)
    for token, index in vocabulary.items():
        tokens[index] = token
    return Tokenizer(
        tokens,
        buffers["merges.txt"].decode("utf-8").split("\n"),
        types,
        config.pre,
        config.special_ids,
    )
