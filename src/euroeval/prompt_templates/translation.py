"""Templates for the Translation task."""

import typing as t

from ..data_models import PromptConfig
from ..languages import DANISH, ENGLISH

if t.TYPE_CHECKING:
    from ..languages import Language

TRANSLATION_TEMPLATES: dict["Language", PromptConfig] = {
    ENGLISH: PromptConfig(
        default_prompt_prefix=(
            "The following are English sentences with Danish translations."
        ),
        default_prompt_template="English: {text}\nDanish: {target_text}",
        default_instruction_prompt=(
            "English: {text}\n\nTranslate the above into Danish. Respond with only "
            "the Danish translation."
        ),
        default_prompt_label_mapping=dict(),
    ),
    DANISH: PromptConfig(
        default_prompt_prefix=(
            "Følgende er engelske sætninger med danske oversættelser."
        ),
        default_prompt_template="Engelsk: {text}\nDansk: {target_text}",
        default_instruction_prompt="Engelsk: {text}\n\nOversæt ovenstående til dansk. "
        "Svar kun med den danske oversættelse.",
        default_prompt_label_mapping=dict(),
    ),
}
