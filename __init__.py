# SMOLFuzz — Synthesizing Models with LLMs for Fuzzing Deep Learning Libraries
#
# Quick-start:
#   from smolfuzz.llm_client import OllamaClient, OpenAIClient, AnthropicClient
#   from smolfuzz.synthesizer import ModelSynthesizer
#   from smolfuzz.executor   import ModelExecutor
#   from smolfuzz.oracle     import DifferentialOracle
#   from smolfuzz.selector   import MultiRouletteSelector
#   from smolfuzz.api_loader import load_and_classify

from .api_loader  import load_and_classify, group_summary
from .llm_client  import LLMBackend, OllamaClient, OpenAIClient, AnthropicClient
from .synthesizer import ModelSynthesizer
from .executor    import ModelExecutor
from .oracle      import DifferentialOracle
from .selector    import MultiRouletteSelector

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
