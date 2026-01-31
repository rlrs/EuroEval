"""Metrics based on LLM-as-a-judge."""

import collections.abc as c
import logging
import typing as t
from dataclasses import replace as dataclasses_replace
from pathlib import Path

import torch
from pydantic import BaseModel, Field, ValidationError

from ..exceptions import InvalidBenchmark
from ..logging_utils import log
from ..tokenisation_utils import apply_chat_template, has_chat_template
from ..utils import create_model_cache_dir, extract_json_dict_from_string
from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig

from ..types import BatchScoringFunction, ScoringFunction


class LLMAsAJudgeMetric(Metric):
    """Use an LLM to judge the quality of the predictions."""

    def __init__(
        self,
        name: str,
        pretty_name: str,
        judge_id: str,
        judge_kwargs: dict[str, t.Any],
        user_prompt: str,
        response_format: t.Type[BaseModel],
        judge_backend: str = "litellm",
        judge_device: str | torch.device | None = None,
        judge_gpu_memory_utilization: float | None = None,
        judge_gpu_memory_gb: float | None = None,
        scoring_fn: ScoringFunction | None = None,
        batch_scoring_fn: BatchScoringFunction | None = None,
        condition_formatting_fn: t.Callable[[str], str] = lambda x: x,
        system_prompt: str | None = None,
    ) -> None:
        """Initialise the LLM as a judge metric.

        Args:
            name:
                The name of the metric in snake_case.
            pretty_name:
                The pretty name of the metric, used for display purposes.
            judge_id:
                The model ID of the LLM to use as a judge.
            judge_kwargs:
                Generation parameters for the judge model, such as temperature.
            user_prompt:
                The user prompt to use for the judge model. The prompt should be
                formatted with the variables `prediction` and `condition`, to
                include the model predictions and a description of what the prediction
                should be judged on, respectively. If the condition is not needed,
                it can be omitted from the prompt, but the `prediction` variable must
                still be present.
            response_format:
                The response format to use for the judge model. This should be a
                Pydantic model that defines the expected structure of the judge's
                response.
            judge_backend:
                The backend to use for the judge model. One of: "litellm", "vllm",
                "transformers".
            judge_device:
                Optional device override for the judge model (e.g., "cpu", "cuda:1").
            judge_gpu_memory_utilization:
                Optional GPU memory utilization override for the judge (vLLM only).
            judge_gpu_memory_gb:
                Optional GPU memory cap in GiB for the judge (vLLM only).
            scoring_fn:
                A function that takes the judge's response and returns a score.
            batch_scoring_fn:
                A function that takes all judge responses and returns a score.
            condition_formatting_fn (optional):
                A function to format the condition string before it is included in the
                user prompt. Defaults to a no-op function that returns the input
                unchanged.
            system_prompt (optional):
                The system prompt to use for the judge model. If not provided, no system
                prompt will be used.
        """
        super().__init__(name=name, pretty_name=pretty_name)
        self.judge_id = judge_id
        self.judge_kwargs = dict(judge_kwargs)
        self.user_prompt = user_prompt
        self.response_format = response_format
        self.judge_backend = judge_backend.lower()
        self.judge_device = judge_device
        self.judge_gpu_memory_utilization = judge_gpu_memory_utilization
        self.judge_gpu_memory_gb = judge_gpu_memory_gb
        self.batch_scoring_fn = self._get_batch_scoring_fn(
            scoring_fn=scoring_fn, batch_scoring_fn=batch_scoring_fn
        )
        self.condition_formatting_fn = condition_formatting_fn
        self.system_prompt = system_prompt

        # Add response format to the generation kwargs
        self._litellm_kwargs = dict(self.judge_kwargs)
        self._litellm_kwargs["response_format"] = self.response_format

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        """Calculate the metric score using the judge model.

        Args:
            predictions:
                The model predictions.
            references:
                The ground truth references.
            dataset:
                The dataset used for evaluation. This is only used in case any
                additional metadata is used to compute the metrics.
            dataset_config:
                The dataset configuration.
            benchmark_config:
                The benchmark configuration.

        Returns:
            The calculated metric score, or None if the score should be ignored.

        Raises:
            InvalidBenchmark:
                If the number of predictions does not match the number of references,
                or if the user prompt requires a condition but none is provided.
        """
        # Importing here to avoid circular imports
        from ..benchmark_modules import LiteLLMModel
        from ..model_cache import ModelCache

        if not predictions or not references:
            return None
        elif len(predictions) != len(references):
            raise InvalidBenchmark(
                f"The number of predictions ({len(predictions):,}) does not match the "
                f"number of references ({len(references):,})."
            )

        judge_benchmark_config = self._get_judge_benchmark_config(
            benchmark_config=benchmark_config
        )

        # Prepare the messages for the LLM
        conversations = [
            [
                dict(
                    role="user",
                    content=self._apply_user_prompt(
                        prediction=prediction, condition=condition
                    ),
                )
            ]
            for prediction, condition in zip(predictions, references)
        ]
        if self.system_prompt:
            conversations = [
                [dict(role="system", content=self.system_prompt), *conversation]
                for conversation in conversations
            ]

        # Load the judge model and cache
        match self.judge_backend:
            case "litellm":
                judge_model_config = LiteLLMModel.get_model_config(
                    model_id=self.judge_id, benchmark_config=judge_benchmark_config
                )
                self.judge = LiteLLMModel(
                    model_config=judge_model_config,
                    dataset_config=dataset_config,
                    benchmark_config=judge_benchmark_config,
                    log_metadata=False,
                    **self._litellm_kwargs,
                )
                judge_cache_dir = Path(judge_model_config.model_cache_dir)
            case "vllm":
                from ..benchmark_modules import VLLMModel

                judge_model_config = VLLMModel.get_model_config(
                    model_id=self.judge_id, benchmark_config=judge_benchmark_config
                )
                judge_dataset_config = self._make_judge_dataset_config(
                    dataset_config=dataset_config
                )
                vllm_kwargs = self._get_generation_kwargs(
                    backend="vllm",
                    default_max_tokens=dataset_config.max_generated_tokens,
                )
                self.judge = VLLMModel(
                    model_config=judge_model_config,
                    dataset_config=judge_dataset_config,
                    benchmark_config=judge_benchmark_config,
                    log_metadata=False,
                    generation_kwargs=vllm_kwargs or None,
                )
                judge_cache_dir = Path(judge_model_config.model_cache_dir)
            case "transformers":
                self.judge = None
                judge_cache_dir = Path(
                    create_model_cache_dir(
                        cache_dir=judge_benchmark_config.cache_dir,
                        model_id=self.judge_id,
                    )
                )
            case _:
                raise InvalidBenchmark(
                    f"Unknown judge backend {self.judge_backend!r}. Expected one of "
                    "'litellm', 'vllm', 'transformers'."
                )

        # Create a cache for the judge model
        judge_cache = ModelCache(
            model_cache_dir=judge_cache_dir,
            cache_name=f"{dataset_config.name}-model-outputs.json",
            max_generated_tokens=dataset_config.max_generated_tokens,
            progress_bar=benchmark_config.progress_bar,
            hash_inputs=not benchmark_config.debug,
        )
        judge_cache.load()

        # Get the non-cached conversations and generate the completions for them
        non_cached_conversations = [
            (idx, conversation)
            for idx, conversation in enumerate(conversations)
            if conversation not in judge_cache
        ]
        if non_cached_conversations:
            if self.judge_backend == "litellm":
                model_inputs = dict(messages=[c for _, c in non_cached_conversations])
                non_cached_outputs = self.judge.generate(inputs=model_inputs)
                judge_cache.add_to_cache(
                    model_inputs=model_inputs, model_output=non_cached_outputs
                )
            elif self.judge_backend == "vllm":
                prompts = [
                    self._render_prompt(
                        conversation=conversation, tokeniser=self.judge._tokeniser
                    )
                    for _, conversation in non_cached_conversations
                ]
                model_inputs = dict(text=prompts)
                non_cached_outputs = self.judge.generate(inputs=model_inputs)
                judge_cache.add_to_cache(
                    model_inputs=dict(
                        messages=[c for _, c in non_cached_conversations]
                    ),
                    model_output=non_cached_outputs,
                )
            elif self.judge_backend == "transformers":
                from ..data_models import GenerativeModelOutput
                from transformers import (
                    AutoConfig,
                    AutoModelForCausalLM,
                    AutoModelForSeq2SeqLM,
                    AutoTokenizer,
                )

                config = AutoConfig.from_pretrained(
                    self.judge_id,
                    cache_dir=judge_benchmark_config.cache_dir,
                    trust_remote_code=judge_benchmark_config.trust_remote_code,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    self.judge_id,
                    cache_dir=judge_benchmark_config.cache_dir,
                    trust_remote_code=judge_benchmark_config.trust_remote_code,
                    use_fast=False,
                )
                if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                prompts = [
                    self._render_prompt(
                        conversation=conversation, tokeniser=tokenizer
                    )
                    for _, conversation in non_cached_conversations
                ]
                if config.is_encoder_decoder:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.judge_id,
                        cache_dir=judge_benchmark_config.cache_dir,
                        trust_remote_code=judge_benchmark_config.trust_remote_code,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.judge_id,
                        cache_dir=judge_benchmark_config.cache_dir,
                        trust_remote_code=judge_benchmark_config.trust_remote_code,
                    )
                model.to(judge_benchmark_config.device)
                model.eval()

                gen_kwargs = self._get_generation_kwargs(
                    backend="transformers",
                    default_max_tokens=dataset_config.max_generated_tokens,
                )
                max_input_length = self._get_transformers_max_input_length(
                    tokenizer=tokenizer,
                    config=config,
                    max_new_tokens=int(gen_kwargs.get("max_new_tokens", 0)),
                )
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=max_input_length is not None,
                    max_length=max_input_length,
                )
                encoded = {
                    k: v.to(judge_benchmark_config.device) for k, v in encoded.items()
                }
                with torch.inference_mode():
                    output_ids = model.generate(**encoded, **gen_kwargs)
                if not config.is_encoder_decoder:
                    prompt_len = encoded["input_ids"].shape[1]
                    output_ids = output_ids[:, prompt_len:]
                sequences = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                non_cached_outputs = GenerativeModelOutput(sequences=sequences)
                judge_cache.add_to_cache(
                    model_inputs=dict(
                        messages=[c for _, c in non_cached_conversations]
                    ),
                    model_output=non_cached_outputs,
                )
            else:
                raise InvalidBenchmark(
                    f"Unknown judge backend {self.judge_backend!r}."
                )
            judge_cache.save()

        # Load all the outputs from the cache, in the original order, and parse them
        raw_outputs = [judge_cache[conversation] for conversation in conversations]
        json_dicts = [
            extract_json_dict_from_string(s=output.sequence) for output in raw_outputs
        ]
        outputs_raw: list[BaseModel | None] = []
        for json_dict in json_dicts:
            if json_dict is None:
                outputs_raw.append(None)
                continue
            try:
                outputs_raw.append(self.response_format.model_validate(obj=json_dict))
            except ValidationError:
                outputs_raw.append(None)

        num_none: int = sum(output is None for output in outputs_raw)
        if num_none:
            log(
                f"Could not parse/validate {num_none:,} of {len(outputs_raw):,} judge "
                f"outputs for metric {self.pretty_name!r}. These will be ignored.",
                level=logging.DEBUG,
            )

        outputs: list[BaseModel] = [
            output for output in outputs_raw if output is not None
        ]
        if not outputs:
            log(
                f"No valid judge outputs were produced for metric "
                f"{self.pretty_name!r}.",
                level=logging.WARNING,
            )
            return None

        return self.batch_scoring_fn(outputs=outputs, dataset=dataset)

    def _apply_user_prompt(self, prediction: str, condition: str | None = None) -> str:
        """Apply the user prompt to the prediction and condition.

        Args:
            prediction:
                The model prediction.
            condition (optional):
                A description of what the prediction should be judged on. If not
                provided, it will be omitted from the prompt.

        Returns:
            The formatted user prompt with the prediction and reference.

        Raises:
            InvalidBenchmark:
                If the user prompt requires a reference but none is provided.
        """
        condition_required = "{condition}" in self.user_prompt
        if condition_required and condition is None:
            raise InvalidBenchmark(
                f"The user prompt for the {self.pretty_name!r} metric requires a "
                "condition, but none was provided."
            )
        if condition is not None:
            return self.user_prompt.format(
                prediction=prediction, condition=self.condition_formatting_fn(condition)
            )
        return self.user_prompt.format(prediction=prediction)

    def _make_judge_dataset_config(
        self, dataset_config: "DatasetConfig"
    ) -> "DatasetConfig":
        """Create a dataset config suitable for judge generation."""
        judge_task = dataclasses_replace(
            dataset_config.task,
            uses_structured_output=False,
            uses_logprobs=False,
            requires_logprobs=False,
        )
        return dataclasses_replace(dataset_config, task=judge_task)

    def _render_prompt(
        self,
        conversation: list[dict[str, str]],
        tokeniser: t.Any | None = None,
    ) -> str:
        """Render a conversation to a prompt string."""
        if tokeniser is not None and has_chat_template(tokeniser=tokeniser):
            return t.cast(
                str,
                apply_chat_template(
                    conversation=conversation,
                    tokeniser=tokeniser,
                    tokenise=False,
                    add_generation_prompt=True,
                ),
            )
        return self._fallback_prompt(conversation=conversation)

    @staticmethod
    def _fallback_prompt(conversation: list[dict[str, str]]) -> str:
        """Fallback prompt rendering when no chat template is available."""
        lines: list[str] = []
        for message in conversation:
            role = message.get("role", "user")
            prefix = role.title()
            lines.append(f"{prefix}: {message.get('content', '')}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _get_generation_kwargs(
        self, backend: str, default_max_tokens: int
    ) -> dict[str, t.Any]:
        """Filter and map generation kwargs for a specific backend."""
        if backend == "vllm":
            allowed = {
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "max_tokens",
                "max_new_tokens",
            }
        elif backend == "transformers":
            allowed = {
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "max_new_tokens",
                "max_tokens",
                "do_sample",
                "num_beams",
            }
        else:
            return {}

        kwargs = {k: v for k, v in self.judge_kwargs.items() if k in allowed}
        if "max_new_tokens" not in kwargs and "max_tokens" not in kwargs:
            kwargs["max_new_tokens"] = default_max_tokens
        elif "max_tokens" in kwargs and "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")
        if "temperature" in kwargs and "do_sample" not in kwargs:
            kwargs["do_sample"] = bool(kwargs["temperature"])
        return kwargs

    def _get_judge_benchmark_config(
        self, benchmark_config: "BenchmarkConfig"
    ) -> "BenchmarkConfig":
        """Create a benchmark config for the judge model if overrides are set."""
        if (
            self.judge_device is None
            and self.judge_gpu_memory_utilization is None
            and self.judge_gpu_memory_gb is None
        ):
            return benchmark_config

        device = benchmark_config.device
        if self.judge_device is not None:
            device = (
                self.judge_device
                if isinstance(self.judge_device, torch.device)
                else torch.device(self.judge_device)
            )

        gpu_memory_utilization = benchmark_config.gpu_memory_utilization
        if self.judge_gpu_memory_gb is not None:
            if device.type != "cuda":
                raise InvalidBenchmark(
                    "judge_gpu_memory_gb requires a CUDA device."
                )
            if not torch.cuda.is_available():
                raise InvalidBenchmark("CUDA is not available.")
            if device.index is None:
                device_index = torch.cuda.current_device()
            else:
                device_index = device.index
            total_bytes = torch.cuda.get_device_properties(device_index).total_memory
            requested_bytes = float(self.judge_gpu_memory_gb) * (1024**3)
            if requested_bytes <= 0:
                raise InvalidBenchmark("judge_gpu_memory_gb must be > 0.")
            if requested_bytes > total_bytes:
                raise InvalidBenchmark(
                    "judge_gpu_memory_gb exceeds available GPU memory."
                )
            gpu_memory_utilization = requested_bytes / total_bytes
        elif self.judge_gpu_memory_utilization is not None:
            gpu_memory_utilization = float(self.judge_gpu_memory_utilization)

        return dataclasses_replace(
            benchmark_config,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    @staticmethod
    def _get_transformers_max_input_length(
        tokenizer: t.Any, config: t.Any, max_new_tokens: int
    ) -> int | None:
        """Compute a safe max input length for transformers generation."""
        candidates: list[int] = []
        model_max_length = getattr(tokenizer, "model_max_length", None)
        if isinstance(model_max_length, int) and model_max_length < 10**9:
            candidates.append(model_max_length)
        max_pos = getattr(config, "max_position_embeddings", None)
        if isinstance(max_pos, int) and max_pos > 0:
            candidates.append(max_pos)
        if not candidates:
            return None

        max_input = min(candidates)
        if getattr(config, "is_encoder_decoder", False):
            return max_input

        reserved = max(1, max_new_tokens)
        return max(1, max_input - reserved)

    def _get_batch_scoring_fn(
        self,
        scoring_fn: ScoringFunction | None,
        batch_scoring_fn: BatchScoringFunction | None,
    ) -> BatchScoringFunction:
        """Get the batch scoring function.

        Args:
            scoring_fn:
                The scoring function to use.
            batch_scoring_fn:
                The batch scoring function to use.

        Returns:
            The batch scoring function.

        Raises:
            InvalidBenchmark:
                If both or neither of the scoring functions are provided.
        """
        if scoring_fn is not None and batch_scoring_fn is not None:
            raise InvalidBenchmark(
                "Both `scoring_fn` and `batch_scoring_fn` are provided. Please "
                "provide only one of them."
            )
        if scoring_fn is not None:
            scoring_fn_nonnull = scoring_fn

            def batch_fn(
                outputs: list[BaseModel], dataset: "Dataset | None" = None
            ) -> float:
                return sum(scoring_fn_nonnull(output) for output in outputs) / len(
                    outputs
                )

            return batch_fn
        if batch_scoring_fn is not None:
            return batch_scoring_fn
        raise InvalidBenchmark(
            "Neither `scoring_fn` nor `batch_scoring_fn` are provided. Please "
            "provide one of them."
        )


### Fluency metric ###


class Fluency(BaseModel):
    """Response format for the fluency metric.

    Attributes:
        fluency:
            The fluency rating, an integer between 1 and 5.
    """

    fluency: t.Annotated[int, Field(ge=1, le=5)]


fluency_metric = LLMAsAJudgeMetric(
    name="fluency",
    pretty_name="Fluency",
    judge_id="gpt-5-2025-08-07",
    judge_kwargs=dict(temperature=1.0),
    user_prompt="Please rate the fluency of the following text on a scale from 1 to 5, "
    "with the following definitions:\n"
    "- 1: Very poor fluency, many grammatical errors\n"
    "- 2: Poor fluency, several grammatical errors\n"
    "- 3: Average fluency, a few grammatical errors\n"
    "- 4: Good fluency, no grammatical errors but sounds a bit off\n"
    "- 5: Excellent fluency, no grammatical errors and sounds natural\n\n"
    "Text: {prediction!r}\n\n"
    "Output your rating as a JSON object with a single key 'fluency'.",
    response_format=Fluency,
    scoring_fn=lambda output: (output.fluency - 1) / 4.0,
)
