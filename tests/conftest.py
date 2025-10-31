"""General fixtures used throughout test modules."""

import collections.abc as c
import os
import sys
from typing import Generator

import pytest
import torch
from click import ParamType

from euroeval.cli import benchmark
from euroeval.data_models import BenchmarkConfig, DatasetConfig, ModelConfig, Task
from euroeval.dataset_configs import get_all_dataset_configs
from euroeval.enums import InferenceBackend, ModelType
from euroeval.languages import DANISH, get_all_languages
from euroeval.metrics import HuggingFaceMetric
from euroeval.tasks import SENT


def pytest_configure() -> None:
    """Set a global flag when `pytest` is being run."""
    setattr(sys, "_called_from_test", True)
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure only one GPU is used in tests


def pytest_unconfigure() -> None:
    """Unset the global flag when `pytest` is finished."""
    delattr(sys, "_called_from_test")


if os.environ.get("CHECK_DATASET") is not None:
    dataset_configs = [
        dataset_config
        for dataset_config in get_all_dataset_configs().values()
        if dataset_config.name in os.environ["CHECK_DATASET"].split(",")
        or any(
            language.code in os.environ["CHECK_DATASET"].split(",")
            for language in dataset_config.languages
        )
        or "all" in os.environ["CHECK_DATASET"].split(",")
    ]
    ACTIVE_LANGUAGES = {
        language_code: language
        for language_code, language in get_all_languages().items()
        if any(language in cfg.languages for cfg in dataset_configs)
    }
else:
    ACTIVE_LANGUAGES = dict(da=DANISH)


@pytest.fixture(scope="session")
def auth() -> Generator[str | bool, None, None]:
    """Yields the authentication token to the Hugging Face Hub."""
    # Get the authentication token to the Hugging Face Hub
    auth = os.environ.get("HUGGINGFACE_API_KEY", True)

    # Ensure that the token does not contain quotes or whitespace
    if isinstance(auth, str):
        auth = auth.strip(" \"'")

    yield auth


@pytest.fixture(scope="session")
def device() -> Generator[torch.device, None, None]:
    """Yields the device to use for the tests."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    yield device


@pytest.fixture(scope="session")
def benchmark_config(
    auth: str, device: torch.device
) -> Generator[BenchmarkConfig, None, None]:
    """Yields a benchmark configuration used in tests."""
    yield BenchmarkConfig(
        model_languages=[DANISH],
        dataset_languages=[DANISH],
        datasets=list(get_all_dataset_configs().values()),
        batch_size=1,
        raise_errors=False,
        cache_dir=".euroeval_cache",
        api_key=auth,
        force=False,
        progress_bar=False,
        save_results=True,
        device=device,
        verbose=False,
        trust_remote_code=True,
        clear_model_cache=False,
        evaluate_test_split=False,
        few_shot=True,
        num_iterations=1,
        api_base=None,
        api_version=None,
        gpu_memory_utilization=0.8,
        generative_type=None,
        debug=False,
        run_with_cli=True,
        requires_safetensors=False,
        download_only=False,
        is_base_model=False,
    )


@pytest.fixture(scope="session")
def metric() -> Generator[HuggingFaceMetric, None, None]:
    """Yields a metric configuration used in tests."""
    yield HuggingFaceMetric(
        name="metric_name",
        pretty_name="Metric name",
        huggingface_id="metric_id",
        results_key="metric_key",
    )


@pytest.fixture(
    scope="session",
    params=list(
        {dataset_config.task for dataset_config in get_all_dataset_configs().values()}
    ),
    ids=lambda task: task.name,
)
def task(request: pytest.FixtureRequest) -> Generator[Task, None, None]:
    """Yields a dataset task used in tests."""
    yield request.param


@pytest.fixture(
    scope="session",
    params=list(ACTIVE_LANGUAGES.values()),
    ids=list(ACTIVE_LANGUAGES.keys()),
)
def language(request: pytest.FixtureRequest) -> Generator[str, None, None]:
    """Yields a language used in tests."""
    yield request.param


@pytest.fixture(scope="session")
def encoder_model_id() -> Generator[str, None, None]:
    """Yields a model ID used in tests."""
    yield "jonfd/electra-small-nordic"


@pytest.fixture(scope="session")
def generative_model_id() -> Generator[str, None, None]:
    """Yields a generative model ID used in tests."""
    yield "HuggingFaceTB/SmolLM2-135M"


@pytest.fixture(scope="session")
def generative_adapter_model_id() -> Generator[str, None, None]:
    """Yields a generative adapter model ID used in tests."""
    yield "jekunz/smollm-135m-lora-fineweb-swedish"


@pytest.fixture(scope="session")
def openai_model_id() -> Generator[str, None, None]:
    """Yields an OpenAI model ID used in tests."""
    yield "gpt-4o-mini"


@pytest.fixture(scope="session")
def ollama_model_id() -> Generator[str, None, None]:
    """Yields an Ollama model ID used in tests."""
    yield "ollama_chat/smollm2:135m"


@pytest.fixture(scope="session")
def model_config() -> Generator[ModelConfig, None, None]:
    """Yields a model configuration used in tests."""
    yield ModelConfig(
        model_id="model_id",
        revision="revision",
        param=None,
        task="task",
        languages=[DANISH],
        merge=False,
        inference_backend=InferenceBackend.TRANSFORMERS,
        model_type=ModelType.ENCODER,
        fresh=True,
        model_cache_dir="cache_dir",
        adapter_base_model_id=None,
    )


@pytest.fixture(scope="module")
def cli_params() -> Generator[dict[str | None, ParamType], None, None]:
    """Yields a dictionary of the CLI parameters."""
    ctx = benchmark.make_context(info_name="testing", args=["--model", "test-model"])
    yield {p.name: p.type for p in benchmark.get_params(ctx)}


@pytest.fixture(scope="session")
def dataset_config() -> c.Generator[DatasetConfig, None, None]:
    """Yields a dataset configuration used in tests."""
    yield DatasetConfig(
        name="dataset",
        pretty_name="Dataset",
        source="dataset_id",
        task=SENT,
        languages=[DANISH],
    )
