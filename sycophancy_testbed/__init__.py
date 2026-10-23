# sycophancy testbed package
# production grade evaluation and closed-loop belief dynamics suite

from .analysis import (
    compute_response_slopes,
    compute_response_surface,
    summarize_static,
    summarize_trajectory,
)
from .client import call_llm, call_llm_with_usage, get_client
from .config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_GEMINI_MODEL,
    SUPPORTED_GROQ_MODELS,
    SUPPORTED_GEMINI_MODELS,
    SUPPORTED_PROVIDERS,
    PROVIDER_MODELS,
    PROVIDER_DEFAULT_MODELS,
    GROQ_BASE_URL,
    OPENROUTER_BASE_URL,
    GEMINI_BASE_URL,
    PILOT_SCENARIOS,
    SCENARIOS,
    SYSTEM_PROMPTS,
    Scenario,
)
from .evaluator import evaluate_response, parse_json_text, validate_evaluation
from .experiments import (
    run_closed_loop,
    run_static_experiment,
    run_static_trial,
    sanity_check_trial,
)
from .metrics import (
    belief_probabilities_from_H,
    compute_h,
    conviction_from_logodds,
    logodds_from_prob,
    short_hash,
    sigmoid,
    stance_from_logodds,
    utc_now,
)
from .plotting import (
    plot_response_surface,
    plot_sycophancy_bars,
    plot_trajectory,
)
from .user_agent import ControlledUser, LLMUser, generate_user_message

__all__ = [
    "Scenario",
    "SCENARIOS",
    "PILOT_SCENARIOS",
    "SYSTEM_PROMPTS",
    "DEFAULT_GROQ_MODEL",
    "DEFAULT_OPENROUTER_MODEL",
    "SUPPORTED_GROQ_MODELS",
    "GROQ_BASE_URL",
    "OPENROUTER_BASE_URL",
    "sigmoid",
    "logodds_from_prob",
    "stance_from_logodds",
    "conviction_from_logodds",
    "compute_h",
    "belief_probabilities_from_H",
    "short_hash",
    "utc_now",
    "get_client",
    "call_llm",
    "call_llm_with_usage",
    "generate_user_message",
    "ControlledUser",
    "LLMUser",
    "evaluate_response",
    "parse_json_text",
    "validate_evaluation",
    "summarize_static",
    "compute_response_slopes",
    "compute_response_surface",
    "summarize_trajectory",
    "plot_sycophancy_bars",
    "plot_response_surface",
    "plot_trajectory",
    "run_static_trial",
    "run_static_experiment",
    "run_closed_loop",
    "sanity_check_trial",
]
