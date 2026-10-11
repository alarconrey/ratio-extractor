"""Only the resolution text and the chosen method are accepted; never a ficha or a label."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from app.settings import MAX_CHARACTERS, MIN_CHARACTERS


class ExtractionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr = Field(min_length=MIN_CHARACTERS, max_length=MAX_CHARACTERS)
    method: Literal["auto", "beto", "tfidf", "rule"] = "auto"

    @field_validator("text")
    @classmethod
    def looks_like_text(cls, value):
        letters = sum(character.isalpha() for character in value)
        if letters < MIN_CHARACTERS // 2:
            raise ValueError("The text must contain mostly letters (a full resolution in HTML or plain text).")
        return value
