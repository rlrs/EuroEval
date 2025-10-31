"""Functions related to text generation of models."""

import collections.abc as c
import logging
import sys
import typing as t
from collections.abc import Mapping, Sequence
from pathlib import Path

import more_itertools as mit
from datasets import Dataset
from tqdm.auto import tqdm

from .enums import BatchingPreference, TaskGroup
from .exceptions import InvalidBenchmark
from .logging_utils import get_pbar, log, log_once
from .model_cache import (
    ModelCache,
    load_cached_model_outputs,
    split_dataset_into_cached_and_non_cached,
)
from .utils import clear_memory

if t.TYPE_CHECKING:
    from datasets import DatasetDict

    from .benchmark_modules import BenchmarkModule
    from .data_models import (
        BenchmarkConfig,
        DatasetConfig,
        GenerativeModelOutput,
        ModelConfig,
    )

logger = logging.getLogger("euroeval")

_generation_observer: t.Callable[..., None] | None = None


def register_generation_observer(observer: t.Callable[..., None] | None) -> None:
    """Register a callback that observes generation outputs.

    Args:
        observer:
            Callable invoked after each generation iteration. Pass ``None`` to clear.
    """

    global _generation_observer
    _generation_observer = observer


def generate(
    model: "BenchmarkModule",
    datasets: c.Sequence["DatasetDict"],
    model_config: "ModelConfig",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
) -> c.Sequence[dict[str, float]]:
    """Evaluate a model on a dataset through generation.

    Args:
        model:
            The model to evaluate.
        datasets:
            The datasets to evaluate on.
        model_config:
            The configuration of the model.
        benchmark_config:
            The configuration of the benchmark.
        dataset_config:
            The configuration of the dataset.

    Returns:
        A list of dictionaries containing the test scores.
    """
    # Set up the name of the model output cache. If we are testing then we save the
    # model outputs to a different cache and ensure that that cache is deleted before
    # the next test, to ensure that the tests are independent of each other
    if benchmark_config.debug:
        model_cache_dir = Path.cwd()
    else:
        model_cache_dir = Path(model_config.model_cache_dir)
    if hasattr(sys, "_called_from_test"):
        cache_name = f"{dataset_config.name}-model-outputs-test.json"
        (model_cache_dir / cache_name).unlink(missing_ok=True)
    elif benchmark_config.debug:
        cache_name = f"{model_config.model_id}-{dataset_config.name}-model-outputs.json"
    else:
        cache_name = f"{dataset_config.name}-model-outputs.json"

    cache = ModelCache(
        model_cache_dir=model_cache_dir,
        cache_name=cache_name,
        max_generated_tokens=dataset_config.max_generated_tokens,
    )

    scores: list[dict[str, float]] = list()
    for idx in get_pbar(
        iterable=range(len(datasets)),
        desc="Benchmarking",
        disable=not benchmark_config.progress_bar,
    ):
        test_scores = generate_single_iteration(
            model=model,
            dataset=datasets[idx]["test"],
            cache=cache,
            dataset_config=dataset_config,
            benchmark_config=benchmark_config,
        )
        log(f"Test scores for iteration {idx}: {test_scores}", level=logging.DEBUG)
        scores.append(test_scores)
        clear_memory()

    if not benchmark_config.debug:
        cache.remove()

    return scores


def generate_single_iteration(
    dataset: "Dataset",
    model: "BenchmarkModule",
    dataset_config: "DatasetConfig",
    benchmark_config: "BenchmarkConfig",
    cache: ModelCache,
) -> dict[str, float]:
    """Evaluate a model on a dataset in a single iteration through generation.

    Args:
        dataset:
            The dataset to evaluate on.
        model:
            The model to evaluate.
        dataset_config:
            The configuration of the dataset.
        benchmark_config:
            The configuration of the benchmark.
        cache:
            The model output cache.

    Returns:
        A list of dictionaries containing the scores for each metric.
    """
    cache.load()

    # Split up the dataset into a cached and non-cached part, unless we are not
    # bootstrapping the samples. In that case, we just use the dataset as is.
    if dataset_config.bootstrap_samples:
        cached_dataset, non_cached_dataset = split_dataset_into_cached_and_non_cached(
            dataset=dataset, cache=cache
        )
    else:
        cached_dataset = Dataset.from_dict({})
        non_cached_dataset = dataset

    all_preds: list[str] = list()

    if len(non_cached_dataset) > 0:
        itr: t.Iterable
        match model.batching_preference:
            case BatchingPreference.SINGLE_SAMPLE:
                itr = get_pbar(iterable=non_cached_dataset)
            case BatchingPreference.ALL_AT_ONCE:
                itr = [non_cached_dataset[:]]
            case _:
                num_batches = len(non_cached_dataset) // benchmark_config.batch_size
                if len(non_cached_dataset) % benchmark_config.batch_size != 0:
                    num_batches += 1
                itr = get_pbar(
                    iterable=mit.batched(
                        iterable=non_cached_dataset, n=benchmark_config.batch_size
                    ),
                    total=len(non_cached_dataset) // benchmark_config.batch_size,
                )

        # Generate the completions for the non-cached examples
        for batch in itr:
            if isinstance(batch, Mapping):
                batch = dict(batch)
            elif isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
                batch_sequence = list(batch)
                if len(batch_sequence) == 0:
                    continue
                if not all(isinstance(sample, Mapping) for sample in batch_sequence):
                    raise TypeError(
                        "Batched dataset samples must be mappings to build model inputs."
                    )
                first_sample = batch_sequence[0]
                batch = {
                    key: [sample[key] for sample in batch_sequence]
                    for key in first_sample.keys()
                }
            else:
                raise TypeError(
                    f"Unsupported batch type {type(batch)!r}; expected mapping or sequence of mappings."
                )

            single_sample_batch = (
                "text" in batch and isinstance(batch["text"], str)
            ) or ("messages" in batch and isinstance(batch["messages"][0], dict))
            if single_sample_batch:
                batch = {key: [value] for key, value in batch.items()}

            model_output = model.generate(inputs=batch)
            extracted_labels = model.extract_labels_from_generation(
                input_batch=batch, model_output=model_output
            )

            # Extended logging if we are running in debug mode
            if benchmark_config.debug:
                debug_log(
                    batch=batch,
                    model_output=model_output,
                    extracted_labels=extracted_labels,  # type: ignore[arg-type]
                    dataset_config=dataset_config,
                )

            cache.add_to_cache(model_inputs=batch, model_output=model_output)
            all_preds.extend(extracted_labels)

            # If we are debugging then we save the cache often, but since this makes
            # evaluation slower, we do not do this by default
            if benchmark_config.debug:
                cache.save()

        if isinstance(itr, tqdm):
            itr.close()

        # Store the cache to disk
        cache.save()

    # Fetch the cached predictions for the cached examples
    if len(cached_dataset) > 0:
        model_output = load_cached_model_outputs(
            cached_dataset=cached_dataset, cache=cache
        )
        extracted_labels = model.extract_labels_from_generation(
            input_batch=cached_dataset[:], model_output=model_output
        )
        all_preds.extend(extracted_labels)

    if "label" in non_cached_dataset.column_names:
        non_cached_labels = non_cached_dataset["label"]
        if not isinstance(non_cached_labels, list):
            non_cached_labels = list(non_cached_labels)
        cached_labels = cached_dataset["label"]
        if not isinstance(cached_labels, list):
            cached_labels = list(cached_labels)
        ground_truth = [
            label.lower() if isinstance(label, str) else label
            for label in non_cached_labels + cached_labels
        ]
    elif "labels" in non_cached_dataset.column_names:
        non_cached_labels = non_cached_dataset["labels"]
        if not isinstance(non_cached_labels, list):
            non_cached_labels = list(non_cached_labels)
        cached_labels = cached_dataset["labels"]
        if not isinstance(cached_labels, list):
            cached_labels = list(cached_labels)
        ground_truth = [
            [label.lower() if isinstance(label, str) else label for label in label_list]
            for label_list in non_cached_labels + cached_labels
        ]
    elif "target_text" in non_cached_dataset.column_names:
        non_cached_labels = non_cached_dataset["target_text"]
        if not isinstance(non_cached_labels, list):
            non_cached_labels = list(non_cached_labels)
        cached_labels = cached_dataset["target_text"]
        if not isinstance(cached_labels, list):
            cached_labels = list(cached_labels)
        ground_truth = non_cached_labels + cached_labels
    else:
        log_once(
            "No labels found in the dataset. We assume that this is intentional, and "
            "will not supply any ground truth labels for evaluation.",
            level=logging.DEBUG,
        )
        ground_truth = []

    itr_scores: dict[str, float] = model.compute_metrics(
        model_outputs_and_labels=(all_preds, ground_truth),
        dataset=dataset,
        benchmark_config=benchmark_config,
    )

    observer = _generation_observer
    if observer is not None:
        try:
            observer(
                dataset_config=dataset_config,
                dataset=dataset,
                benchmark_config=benchmark_config,
                predictions=all_preds,
                references=ground_truth,
            )
        except Exception:  # pragma: no cover - observer errors should not break eval
            logger.debug(
                "generation_observer_failed",
                exc_info=True,
            )

    return itr_scores


def debug_log(
    batch: dict[str, t.Any],
    model_output: "GenerativeModelOutput",
    extracted_labels: c.Sequence[dict | str | c.Sequence[str]],
    dataset_config: "DatasetConfig",
) -> None:
    """Log inputs and outputs for debugging purposes.

    Args:
        batch:
            The batch of examples to evaluate on.
        model_output:
            The output of the model.
        extracted_labels:
            The extracted labels from the model output.
        dataset_config:
            The configuration of the dataset.
    """
    match dataset_config.task.task_group:
        case TaskGroup.TOKEN_CLASSIFICATION:
            log_msgs = [""]
            for tokens, predictions, labels in zip(
                batch["tokens"], extracted_labels, batch["labels"]
            ):
                predictions = [tag.upper() for tag in predictions]
                sample = list(zip(tokens, predictions, labels))
                log_batches = [
                    [("Tokens: ", "Predictions: ", "Labels: ")] + sample[i : i + 10]
                    for i in range(0, len(sample), 10)
                ]
                for log_batch in log_batches:
                    lengths = [len(max(triple, key=len)) for triple in log_batch]
                    log_batch = [
                        [f"{x:<{length}}" for x in triple]
                        for triple, length in zip(log_batch, lengths)
                    ]
                    tokens = [triple[0] for triple in log_batch]
                    predictions = [triple[1] for triple in log_batch]
                    labels = [triple[2] for triple in log_batch]
                    log_msgs.append(
                        "\t".join(tokens)
                        + "\n"
                        + "\t".join(predictions)
                        + "\n"
                        + "\t".join(labels)
                    )
            log("\n\n".join(log_msgs), level=logging.DEBUG)
            return

        case (
            TaskGroup.SEQUENCE_CLASSIFICATION | TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION
        ):
            if "label" in batch:
                labels = [
                    dataset_config.prompt_label_mapping.get(label, label).lower()
                    for label in batch["label"]
                ]
            else:
                labels = [None] * len(extracted_labels)

        case TaskGroup.QUESTION_ANSWERING:
            extracted_labels = [
                prediction["prediction_text"]
                for prediction in extracted_labels
                if isinstance(prediction, dict)
            ]
            labels = [label["answers"]["text"][0] for label in batch["label"]]

        case TaskGroup.TEXT_TO_TEXT:
            labels = batch["target_text"]

        case _:
            raise InvalidBenchmark(
                f"The task group '{dataset_config.task.task_group}' is not supported."
            )

    if "messages" in batch:
        input_texts = [messages[-1]["content"] for messages in batch["messages"]]
    else:
        input_texts = batch["text"]

    metadata_keys: c.Sequence[str] = [
        key
        for key in batch.keys()
        if key not in ["text", "messages", "label", "labels", "target_text"]
    ]

    for idx in range(len(input_texts)):
        data_to_log: dict[str, t.Any] = {
            "Input": input_texts[idx],
            "Raw output": model_output.sequences[idx],
            "Prediction": extracted_labels[idx],
        }
        if labels[idx]:
            data_to_log["Label"] = labels[idx]
        data_to_log |= {key.capitalize(): batch[key][idx] for key in metadata_keys}
        log(
            "\n".join(f"{key}: {value!r}" for key, value in data_to_log.items()),
            level=logging.DEBUG,
        )
