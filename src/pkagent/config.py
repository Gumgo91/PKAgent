"""Settings of an analysis session: LLM, budgets and numerical options."""
from dataclasses import dataclass, field
from pathlib import Path
import os

OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
MODELS = {                                   # short names used on the command line
    'gpt': 'openai/gpt-6.1-sol',
    'claude': 'anthropic/claude-opus-5.5',
}


def load_env(path=None):
    """Read KEY=VALUE lines of a .env file into os.environ (existing variables win): the given path, else .env in
    the working directory, else .env in the project root (next to src/)."""
    candidates = [Path(path)] if path else [Path('.env'), Path(__file__).resolve().parents[2] / '.env']
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        return
    for line in p.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def api_key():
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise RuntimeError('PKAgent needs an OpenRouter API key: set OPENROUTER_API_KEY in the environment or in .env')
    return key


@dataclass
class Budget:
    """Limits of one session; the agent sees the remaining budget after every tool call."""
    max_turns: int = 80              # LLM responses
    max_fits: int = 40               # model fits (a stepwise covariate search counts its fits)
    max_hours: float = 6.
    max_cost_usd: float = 25.


@dataclass
class Settings:
    model: str = MODELS['claude']
    temperature: float | None = None     # provider default when None
    reasoning_effort: str | None = 'medium'
    max_output_tokens: int = 8000
    budget: Budget = field(default_factory=Budget)
    seed: int = 20261001
    workers: int = 2                     # parallel fitting processes
    threads_per_worker: int = 4          # Numba threads in each fitting process
    fit_wall_seconds: float = 1800.      # wall-clock budget of one fit (Laplace minimization or importance refinement)
    laplace_starts: int = 1              # starts of the Laplace minimization (the agent's values, then perturbations)
    laplace_seconds: float = 300.        # Laplace exploration budget before an importance refinement
    laplace_tolerance: float = .5        # a Laplace fit is converged when its Newton decrement is at most this (OFV)
    images: bool = True                  # let the agent look at diagnostic plots
