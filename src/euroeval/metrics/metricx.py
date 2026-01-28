"""MetricX metric implementation."""

from __future__ import annotations

import collections.abc as c
import logging
import typing as t
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, MT5Config
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration

from ..exceptions import InvalidBenchmark
from ..logging_utils import log, log_once
from .base import Metric

if t.TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset

    from ..data_models import BenchmarkConfig, DatasetConfig


@dataclass
class MT5ForRegressionOutput:
    """Output for the MT5 regression model."""

    predictions: torch.Tensor
    encoder_last_hidden_state: torch.Tensor


class MT5ForRegression(MT5ForConditionalGeneration):
    """MT5 regression model used by MetricX."""

    def __init__(self, config: MT5Config) -> None:
        super().__init__(config)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
    ) -> MT5ForRegressionOutput:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)
        preds = lm_logits[:, 0, 250089]
        preds = torch.clamp(preds, min=0, max=25)

        return MT5ForRegressionOutput(
            predictions=preds,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )


class MetricXMetric(Metric):
    """MetricX scorer using the MT5 regression model."""

    def __init__(
        self,
        name: str,
        pretty_name: str,
        model_id: str,
        tokenizer_id: str = "google/mt5-xl",
        max_length: int = 1536,
        batch_size: int = 4,
        use_bf16: bool = True,
        postprocessing_fn: t.Callable[[float], tuple[float, str]] | None = None,
    ) -> None:
        super().__init__(
            name=name, pretty_name=pretty_name, postprocessing_fn=postprocessing_fn
        )
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_bf16 = use_bf16
        self.model: MT5ForRegression | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.device: torch.device | None = None
        self._warned_truncation = False

    def _ensure_loaded(self, benchmark_config: "BenchmarkConfig") -> None:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_id,
                use_fast=False,
                legacy=False,
                cache_dir=benchmark_config.cache_dir,
            )
        if self.model is None:
            dtype = None
            if (
                self.use_bf16
                and benchmark_config.device.type == "cuda"
                and torch.cuda.is_bf16_supported()
            ):
                dtype = torch.bfloat16
            self.model = MT5ForRegression.from_pretrained(
                self.model_id, cache_dir=benchmark_config.cache_dir, torch_dtype=dtype
            )
            self.model.eval()

        if self.device != benchmark_config.device:
            self.device = benchmark_config.device
            assert self.model is not None
            self.model.to(self.device)

    def _encode_sample(
        self, source: str, hypothesis: str, reference: str
    ) -> list[int]:
        assert self.tokenizer is not None
        source = "source: " + source
        hypothesis = " candidate: " + hypothesis
        reference = " reference: " + reference

        tokens_source = self.tokenizer.encode(
            source, add_special_tokens=False, truncation=False
        )
        tokens_hyp = self.tokenizer.encode(
            hypothesis, add_special_tokens=False, truncation=False
        )
        tokens_ref = self.tokenizer.encode(
            reference, add_special_tokens=False, truncation=False
        )
        input_ids = tokens_source + tokens_hyp + tokens_ref
        if len(input_ids) > self.max_length and not self._warned_truncation:
            log_once(
                "MetricX input exceeds max length; truncation will be applied.",
                level=logging.WARNING,
            )
            self._warned_truncation = True
        prepared = self.tokenizer.prepare_for_model(
            input_ids,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
        )
        return prepared["input_ids"]

    def _score_batch(
        self, input_ids_list: list[list[int]]
    ) -> torch.Tensor:
        assert self.tokenizer is not None
        assert self.model is not None
        assert self.device is not None

        batch = self.tokenizer.pad(
            {"input_ids": input_ids_list}, return_tensors="pt"
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            outputs = self.model(**batch)
        return outputs.predictions

    def __call__(
        self,
        predictions: c.Sequence,
        references: c.Sequence,
        dataset: "Dataset",
        dataset_config: "DatasetConfig",
        benchmark_config: "BenchmarkConfig",
    ) -> float | None:
        if dataset is None:
            raise InvalidBenchmark("MetricX requires the dataset to obtain sources.")

        sources = dataset["text"]
        if not len(sources) == len(predictions) == len(references):
            raise InvalidBenchmark(
                "MetricX expects sources, predictions, and references to be the same "
                f"length, got {len(sources)}, {len(predictions)}, and {len(references)}."
            )

        self._ensure_loaded(benchmark_config=benchmark_config)

        input_ids_list = [
            self._encode_sample(source=src, hypothesis=pred, reference=ref)
            for src, pred, ref in zip(sources, predictions, references)
        ]

        all_scores: list[torch.Tensor] = []
        for i in range(0, len(input_ids_list), self.batch_size):
            batch_input_ids = input_ids_list[i : i + self.batch_size]
            try:
                scores = self._score_batch(batch_input_ids)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise InvalidBenchmark(str(e)) from e
                if self.device is not None and self.device.type == "cuda":
                    log(
                        "MetricX ran out of memory on GPU; retrying on CPU.",
                        level=logging.WARNING,
                    )
                    assert self.model is not None
                    self.model = self.model.to("cpu")
                    self.device = torch.device("cpu")
                    scores = self._score_batch(batch_input_ids)
                else:
                    raise InvalidBenchmark(str(e)) from e
            all_scores.append(scores.detach().cpu())

        mean_score = torch.cat(all_scores).mean().item()
        return float(mean_score)


metricx_24_hybrid_xxl_metric = MetricXMetric(
    name="metricx_24_hybrid_xxl",
    pretty_name="MetricX-24 Hybrid XXL",
    model_id="google/metricx-24-hybrid-xxl-v2p6-bfloat16",
    postprocessing_fn=lambda x: (x, f"{x:.4f}"),
)
