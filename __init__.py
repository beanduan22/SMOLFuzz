# SMOLFuzz — Synthesizing Models with LLMs for Fuzzing Deep Learning Libraries
#
# Quick-start:
#   from smolfuzz.backends.llm_client import OllamaClient, OpenAIClient, AnthropicClient
#   from smolfuzz.core.synthesizer    import ModelSynthesizer
#   from smolfuzz.core.executor       import ModelExecutor
#   from smolfuzz.core.oracle         import DifferentialOracle
#   from smolfuzz.core.selector       import MultiRouletteSelector
#   from smolfuzz.core.api_loader     import load_and_classify

from .core.api_loader   import load_and_classify, group_summary
from .backends.llm_client import LLMBackend, OllamaClient, OpenAIClient, AnthropicClient
from .core.synthesizer  import ModelSynthesizer
from .core.executor     import ModelExecutor
from .core.oracle       import DifferentialOracle
from .core.selector     import MultiRouletteSelector

__all__ = [
    "LLMBackend",
    "OllamaClient",
    "OpenAIClient",
    "AnthropicClient",
    "ModelSynthesizer",
    "ModelExecutor",
    "DifferentialOracle",
    "MultiRouletteSelector",
    "load_and_classify",
    "group_summary",
]
