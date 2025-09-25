"""The different types of modules that can be benchmarked."""

from .base import BenchmarkModule
from .fresh import FreshEncoderModel
from .gateway import GatewayModel
from .hf import HuggingFaceEncoderModel
from .litellm import LiteLLMModel
from .vllm import VLLMModel
