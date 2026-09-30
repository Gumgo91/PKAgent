"""PKAgent: an LLM agent that develops population PK/PD models with the PKPy2 engine."""
__version__ = '2.0.0.dev0'

from .agent import run                     # noqa: E402,F401
from .config import MODELS, Budget, Settings   # noqa: E402,F401
