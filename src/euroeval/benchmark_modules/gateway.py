"""Gateway-backed benchmark module.

This module bridges EuroEval with the llm-gateway orchestration layer. When the
`EUROEVAL_GATEWAY_MODE` environment variable is set it will either record the
requests that EuroEval would issue (phase 1) or replay pre-recorded responses to
produce EuroEval-compatible outputs (phase 2).
"""

from __future__ import annotations

import json
import logging
import os
import typing as t
from functools import lru_cache, partial
from pathlib import Path

from datasets import DatasetDict

from ..constants import (
    JSON_STRIP_CHARACTERS,
    LITELLM_CLASSIFICATION_OUTPUT_KEY,
    MAX_LITELLM_LOGPROBS,
    REASONING_MAX_TOKENS,
)
from ..data_models import BenchmarkConfig, DatasetConfig, GenerativeModelOutput, ModelConfig
from ..enums import BatchingPreference, GenerativeType, InferenceBackend, ModelType, TaskGroup
from ..exceptions import InvalidBenchmark
from ..generation_utils import apply_prompt, extract_few_shot_examples
from ..task_group_utils import (
    question_answering,
    sequence_classification,
    text_to_text,
    token_classification,
)
from ..types import ExtractLabelsFunction
from ..utils import create_model_cache_dir, log_once, split_model_id
from .base import BenchmarkModule

if t.TYPE_CHECKING:
    from transformers.trainer import Trainer
    from ..data_models import Task

logger = logging.getLogger("euroeval")


def _strip_trailing_prompt_whitespace(batch: dict[str, t.Any]) -> dict[str, t.Any]:
    """Remove trailing spaces/tabs from prompt-related fields.

    Few-shot exemplars end in tokens like " ja", but the actual example to predict
    on previously had "Grammatisk korrekt: " with a trailing space. This caused the
    tokenizer to see a different prefix for the final example compared to the few-shot
    answers. Trimming spaces and tabs keeps the colon while letting the model generate
    the leading-space token consistently.
    """

    def _clean(values: list[t.Any]) -> list[t.Any]:
        return [value.rstrip(" \t") if isinstance(value, str) else value for value in values]

    for field in ("text", "prompt"):
        if field in batch and batch[field] is not None:
            batch[field] = _clean(batch[field])
    return batch


def _normalise_response_text(text: str, *, uses_structured_output: bool) -> str:
    """Normalise model responses before downstream parsing.

    Token-classification datasets expect JSON output; leading whitespace here can cause
    the JSON extractor to fail, so we only trim whitespace characters in that case.
    Other tasks historically rely on `JSON_STRIP_CHARACTERS` to tame various wrappers,
    so we keep that behaviour when structured output is not used.
    """

    return text.strip() if uses_structured_output else text.strip(JSON_STRIP_CHARACTERS)


class GatewayModel(BenchmarkModule):
    """Benchmark module that records requests or replays responses."""

    fresh_model = False
    batching_preference = BatchingPreference.SINGLE_SAMPLE
    high_priority = True

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        benchmark_config: BenchmarkConfig,
        log_metadata: bool = True,
        **generation_kwargs: t.Any,
    ) -> None:
        mode = os.getenv("EUROEVAL_GATEWAY_MODE")
        if not mode:
            raise InvalidBenchmark(
                "GatewayModel requires EUROEVAL_GATEWAY_MODE to be set to 'phase1' or 'phase2'."
            )
        mode = mode.lower()
        if mode not in {"phase1", "phase2"}:
            raise InvalidBenchmark(
                f"Unsupported EUROEVAL_GATEWAY_MODE value: {mode!r}. Expected 'phase1' or 'phase2'."
            )

        self.mode = mode
        env_is_base = os.getenv("EUROEVAL_IS_BASE_MODEL", "").lower() == "true"
        self._is_base_model = benchmark_config.is_base_model or env_is_base
        self._manual_generation_kwargs = generation_kwargs

        self._requests_path: Path | None = None
        self._responses_path: Path | None = None
        self._responses: list[dict[str, t.Any]] = []
        self._response_cursor = 0

        if self.mode == "phase1":
            requests_env = os.getenv("EUROEVAL_GATEWAY_REQUESTS_FILE")
            if not requests_env:
                raise InvalidBenchmark(
                    "EUROEVAL_GATEWAY_REQUESTS_FILE must be set in phase1 mode."
                )
            self._requests_path = Path(requests_env)
        else:
            responses_env = os.getenv("EUROEVAL_GATEWAY_RESPONSES_FILE")
            if not responses_env:
                raise InvalidBenchmark(
                    "EUROEVAL_GATEWAY_RESPONSES_FILE must be set in phase2 mode."
                )
            self._responses_path = Path(responses_env)
            if self._responses_path.exists():
                with self._responses_path.open("r", encoding="utf-8") as fh:
                    self._responses = [
                        json.loads(line)
                        for line in fh
                        if line.strip()
                    ]
            else:
                log_once(
                    f"Responses file {self._responses_path} does not exist; proceeding with empty responses.",
                    level=logging.WARNING,
                )

        super().__init__(
            model_config=model_config,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
            log_metadata=log_metadata,
        )

    # ------------------------------------------------------------------
    # BenchmarkModule abstract API implementations
    # ------------------------------------------------------------------
    def prepare_dataset(
        self, dataset: DatasetDict, task: "Task", itr_idx: int
    ) -> DatasetDict:
        if task.task_group == TaskGroup.QUESTION_ANSWERING:
            dataset = dataset.map(
                lambda examples: dict(
                    label=[
                        dict(
                            id=id,
                            answers=dict(
                                answer_start=answer_dct["answer_start"],
                                text=[answer_text.lower() for answer_text in answer_dct["text"]],
                            ),
                        )
                        for id, answer_dct in zip(examples["id"], examples["answers"])
                    ]
                ),
                batched=True,
                load_from_cache_file=False,
                keep_in_memory=True,
            )

        if self.benchmark_config.few_shot:
            few_shot_examples = extract_few_shot_examples(
                dataset=dataset,
                dataset_config=self.dataset_config,
                benchmark_config=self.benchmark_config,
                itr_idx=itr_idx,
            )
        else:
            few_shot_examples = list()

        dataset["test"] = dataset["test"].map(
            partial(
                apply_prompt,
                few_shot_examples=few_shot_examples,
                model_config=self.model_config,
                dataset_config=self.dataset_config,
                generative_type=self.generative_type,
                always_populate_text_field=False,
                tokeniser=None,
            ),
            batched=True,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

        dataset["test"] = dataset["test"].map(
            _strip_trailing_prompt_whitespace,
            batched=True,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

        return dataset

    def generate(self, inputs: dict) -> GenerativeModelOutput:
        samples = self._extract_samples(inputs)

        if self.mode == "phase1":
            for idx, sample in enumerate(samples):
                payload = self._build_request_payload(sample)
                self._record_request(payload)
            return GenerativeModelOutput(sequences=["" for _ in samples], scores=None)

        raw_sequences: list[str] = []
        scores: list[list[list[tuple[str, float]]]] = []
        for idx, sample in enumerate(samples):
            response = self._next_response()
            content = response.get("content") or ""
            raw_sequences.append(str(content))

            raw_logprobs = response.get("logprobs")
            logprob_list = self._convert_logprobs(raw_logprobs)
            scores.append(logprob_list)

        sequences = self._postprocess_sequences(raw_sequences)
        use_scores = None
        if any(score for score in scores):
            use_scores = scores

        return GenerativeModelOutput(sequences=sequences, scores=use_scores)

    @classmethod
    def model_exists(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> bool:
        return os.getenv("EUROEVAL_GATEWAY_MODE") is not None

    @classmethod
    def get_model_config(
        cls, model_id: str, benchmark_config: BenchmarkConfig
    ) -> ModelConfig:
        model_id_components = split_model_id(model_id=model_id)
        cache_dir = create_model_cache_dir(
            cache_dir=benchmark_config.cache_dir,
            model_id=f"gateway/{model_id_components.model_id}"
        )
        return ModelConfig(
            model_id=model_id_components.model_id,
            revision=model_id_components.revision,
            param=model_id_components.param,
            task="text-generation",
            languages=list(),
            inference_backend=InferenceBackend.GATEWAY,
            merge=False,
            model_type=ModelType.GENERATIVE,
            fresh=False,
            model_cache_dir=cache_dir,
            adapter_base_model_id=None,
        )

    @property
    def generative_type(self) -> GenerativeType:
        if self.benchmark_config.generative_type is not None:
            return self.benchmark_config.generative_type
        if self._is_base_model:
            return GenerativeType.BASE
        return GenerativeType.INSTRUCTION_TUNED

    @property
    def data_collator(self) -> t.Callable[[list[t.Any]], dict[str, t.Any]]:
        raise NotImplementedError(
            "The `data_collator` property is not implemented for GatewayModel."
        )

    @property
    def extract_labels_from_generation(self) -> ExtractLabelsFunction:
        match self.dataset_config.task.task_group:
            case (
                TaskGroup.SEQUENCE_CLASSIFICATION
                | TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
            ):
                wants_logprobs = self.dataset_config.task.uses_logprobs
                return partial(
                    sequence_classification.extract_labels_from_generation,
                    dataset_config=self.dataset_config,
                    first_label_token_mapping=True if wants_logprobs else False,
                )
            case TaskGroup.TEXT_TO_TEXT:
                return text_to_text.extract_labels_from_generation
            case TaskGroup.TOKEN_CLASSIFICATION:
                return partial(
                    token_classification.extract_labels_from_generation,
                    dataset_config=self.dataset_config,
                )
            case TaskGroup.QUESTION_ANSWERING:
                return question_answering.extract_labels_from_generation
            case _:
                raise NotImplementedError(
                    f"Unsupported task group: {self.dataset_config.task.task_group}."
                )

    @property
    def trainer_class(self) -> t.Type["Trainer"]:
        raise NotImplementedError(
            "The `trainer_class` property is not implemented for GatewayModel."
        )

    @property
    def num_params(self) -> int:
        return -1

    @property
    def vocab_size(self) -> int:
        return -1

    @property
    def model_max_length(self) -> int:
        if self.generative_type == GenerativeType.REASONING:
            return REASONING_MAX_TOKENS
        return self.dataset_config.max_generated_tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_samples(self, inputs: dict) -> list[dict[str, t.Any]]:
        messages = inputs.get("messages")
        prompts = inputs.get("prompt")
        texts = inputs.get("text")

        if messages is not None:
            sample_count = len(messages)
        elif texts is not None:
            sample_count = len(texts)
        elif prompts is not None:
            sample_count = len(prompts)
        else:
            raise InvalidBenchmark(
                "GatewayModel expects inputs to contain 'messages', 'prompt', or 'text'."
            )

        samples: list[dict[str, t.Any]] = []
        for idx in range(sample_count):
            sample: dict[str, t.Any] = {}

            if messages is not None:
                sample["messages"] = messages[idx]

            if self.generative_type == GenerativeType.BASE:
                if texts is not None:
                    sample["prompt"] = texts[idx]
                    if prompts is not None:
                        sample["_raw_prompt"] = prompts[idx]
                elif prompts is not None:
                    sample["prompt"] = prompts[idx]
                else:
                    raise InvalidBenchmark(
                        "Base models require 'text' or 'prompt' inputs for generation."
                    )
            else:
                if prompts is not None:
                    sample["prompt"] = prompts[idx]
                elif texts is not None:
                    sample["prompt"] = texts[idx]

            samples.append(sample)

        return samples

    def _build_request_payload(self, sample: dict[str, t.Any]) -> dict[str, t.Any]:
        payload: dict[str, t.Any] = {
            "generation_kwargs": dict(self._effective_generation_kwargs()),
            "dataset": self.dataset_config.name,
        }
        if "messages" in sample and sample["messages"] is not None:
            payload["messages"] = sample["messages"]
        if "prompt" in sample and sample["prompt"] is not None:
            payload["prompt"] = sample["prompt"]
        return payload

    def _record_request(self, payload: dict[str, t.Any]) -> None:
        if not self._requests_path:
            return
        self._requests_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False)
        with self._requests_path.open("a", encoding="utf-8") as fh:
            fh.write(serialized + "\n")

    def _next_response(self) -> dict[str, t.Any]:
        if self._response_cursor >= len(self._responses):
            raise InvalidBenchmark(
                "Not enough responses provided to replay EuroEval generation."
            )
        response = self._responses[self._response_cursor]
        self._response_cursor += 1
        return response

    def _convert_logprobs(
        self, raw_logprobs: t.Any
    ) -> list[list[tuple[str, float]]]:
        if not raw_logprobs:
            return []

        try:
            if isinstance(raw_logprobs, dict):
                if "content" in raw_logprobs and isinstance(raw_logprobs["content"], list):
                    converted: list[list[tuple[str, float]]] = []
                    for entry in raw_logprobs["content"]:
                        if not isinstance(entry, dict):
                            continue
                        top = entry.get("top_logprobs") or []
                        if top:
                            converted.append(
                                [
                                    (str(item.get("token", "")), float(item.get("logprob", 0.0)))
                                    for item in top
                                    if item is not None
                                ]
                            )
                        else:
                            token = str(entry.get("token", ""))
                            logprob = float(entry.get("logprob", 0.0))
                            converted.append([(token, logprob)])
                    return converted
                if "top_logprobs" in raw_logprobs and isinstance(raw_logprobs["top_logprobs"], list):
                    return [
                        [
                            (str(token), float(logprob))
                            for token, logprob in (entry or dict()).items()
                        ]
                        for entry in raw_logprobs["top_logprobs"]
                    ]
        except Exception as exc:  # pragma: no cover - defensive logging
            log_once(
                f"Failed to convert logprobs payload: {exc}. Falling back to empty scores.",
                level=logging.DEBUG,
            )
        return []

    @lru_cache
    def _effective_generation_kwargs(self) -> dict[str, t.Any]:
        if self._manual_generation_kwargs:
            return dict(self._manual_generation_kwargs)

        max_tokens = (
            REASONING_MAX_TOKENS
            if self.generative_type == GenerativeType.REASONING
            else self.dataset_config.max_generated_tokens
        )
        stop_tokens: list[str] = []
        if self.generative_type == GenerativeType.BASE:
            stop_tokens.append("\n\n")
        kwargs: dict[str, t.Any] = {
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "stop": stop_tokens,
            "seed": 4242,
        }
        if self.dataset_config.task.uses_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = MAX_LITELLM_LOGPROBS
        return kwargs

    def _log_metadata(self) -> None:
        # Override parent logging to avoid querying external services for metadata.
        if not self.log_metadata:
            return
        log_once(
            "Using gateway backend; model metadata (params/vocab/context) unavailable.",
            level=logging.INFO,
        )

    def _handle_classification_json(self, content: str) -> str:
        if LITELLM_CLASSIFICATION_OUTPUT_KEY in content:
            try:
                payload = json.loads(content)
                if (
                    isinstance(payload, dict)
                    and set(payload.keys()) == {LITELLM_CLASSIFICATION_OUTPUT_KEY}
                ):
                    return str(payload[LITELLM_CLASSIFICATION_OUTPUT_KEY]).strip()
            except json.JSONDecodeError:
                pass
        return content

    def _postprocess_sequences(self, sequences: list[str]) -> list[str]:
        processed: list[str] = []
        for sequence in sequences:
            raw_text = str(sequence)
            text = self._handle_classification_json(raw_text)
            processed.append(
                _normalise_response_text(
                    text,
                    uses_structured_output=self.dataset_config.task.uses_structured_output,
                )
            )
        return processed
