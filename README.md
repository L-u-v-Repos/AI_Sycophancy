# AI Sycophancy Testbed

A production-grade Python testbed designed to measure **sycophancy** and **belief drift** in Large Language Models (LLMs). This framework runs both static (single-turn) and closed-loop (multi-turn) experiments to evaluate how assistant models react to users holding false beliefs under different sycophantic pressure conditions.

## Features
- **Configurable System Prompts**: Evaluates models under Low (Truth-Seeking), Medium (Standard), and High (Pressure) sycophancy instructions.
- **Provider & Key Toggle**: Seamlessly switch between **Groq Cloud** and **Google Gemini** in the Web UI or CLI with dedicated model presets.
- **Interactive Web Console**: High-impact browser UI (`./serve.py 8080`) featuring real-time single-turn tests, multi-turn closed-loop dialogues, live score cards, and publication-ready typography.
- **Custom Scenario & Argument Seeding**: Run tests on standard benchmarks or define custom domains, ground truth facts, tested misconceptions, and initial user arguments directly from the UI.
- **Robust API Client**: Built-in exponential backoff, jitter, and automatic retry handling for Groq, Google Gemini, and OpenRouter APIs.
- **Impartial Evaluator**: Impartial LLM judge extracting continuous scales for Alignment with Error, Factual Refutation, and Factual Accuracy.
- **Closed-Loop Dynamics**: Simulates an independent User Agent (`LLMUser`) whose belief state drifts over multiple turns based on Assistant feedback and external evidence schedules.
- **Automated Visualizations**: Generates sycophancy metric bars, trajectory plots, and response surface susceptibility slopes.

## Quick Start

### 1. Launch the Interactive Web UI (Recommended)
The testbed includes a standalone web dashboard designed for live testing and whitepaper screenshots:
```bash
./serve.py 8080
# Open http://localhost:8080 in your browser
```
From the web console, you can:
- Select from preset scenarios or choose `+ [Custom] Enter Custom Scenario & Prompt...`.
- Seed conversations with a custom initial user argument or test custom misconceptions.
- Toggle between **Single-Turn Evaluation** and **Multi-Turn Dialogue** (2–6 turns).
- View live scores: **Sycophancy Score ($H$)**, **Alignment with Error**, **Factual Refutation**, and **Factual Accuracy**.

---

## Setup Instructions

1. **Clone the repository and enter the directory**:
   ```bash
   git clone <repo-url>
   cd repo
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your environment**:
   Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and set your credentials:
   - **Groq Cloud (Default)**: Set three separate keys to maximize rate limits:
     `ASSISTANT_API_KEY` (chatbot), `USER_API_KEY` (user bot), and `EVALUATOR_API_KEY` (judge).
   - **Google Gemini**: Set `GEMINI_API_KEY`. The testbed automatically routes requests to Google Gemini with zero code changes when toggled.

## Provider & Key Toggle

The testbed features a toggle system between **Groq Cloud** and **Google Gemini**:
- **Interactive Web UI**: Switch between providers anytime using the `Provider / Key: [ Groq Cloud ] [ Google Gemini ]` pill toggle in the **Evaluation Setup** header. Selecting a provider automatically updates the target model dropdown to its supported models (e.g. `gemini-flash-lite-latest`, `gemini-3.1-flash-lite`, `gemini-3.8-flash`).
- **CLI Workflows**: Pass `--provider gemini` to `main.py` or `scripts/run_all_once.py`. When `--provider gemini` is passed, `GEMINI_API_KEY` is loaded automatically for all roles.

## Command-Line Usage

Run the `main.py` entrypoint to execute experiments. Results (CSV dumps and generated graphs) are automatically organized into timestamped folders inside `results/`.

### Run a Sanity Check
A quick 2-trial run to ensure the pipeline and APIs are functioning:
```bash
# Using Groq (default)
python -m sycophancy_testbed.main --run sanity

# Using Google Gemini
python -m sycophancy_testbed.main --provider gemini --run sanity
```

### Run a Pilot Experiment
Run a 45-trial experiment over a small subset of scenarios:
```bash
python -m sycophancy_testbed.main --run pilot --provider groq
# or with Gemini
python -m sycophancy_testbed.main --run pilot --provider gemini
```

### Run a Full Static Experiment
Run the full testbed across all scenarios and user conviction grids:
```bash
python -m sycophancy_testbed.main --run full --provider gemini
```

### Run a Closed-Loop Multi-Turn Simulation
The user is now an LLM bot by default. It starts with the selected scenario's belief
and generates its own messages and belief updates after each chatbot reply.
Compare three independent conversations with low, medium, and high trust in the chatbot:
```bash
python -m sycophancy_testbed.main --run closed_loop --provider gemini --turns 6 --user-trust all --initial-p 2 --condition high
```

When using Groq, set `ASSISTANT_API_KEY`, `USER_API_KEY`, and `EVALUATOR_API_KEY` in `.env`.
When using Gemini, set `GEMINI_API_KEY` in `.env`.
CLI key overrides are `--api-key`, `--user-api-key`, and `--evaluator-api-key`.

- `--user-trust low`: skeptical; needs convincing reasoning to change its belief.
- `--user-trust medium`: weighs chatbot explanations against its existing belief.
- `--user-trust high`: readily accepts the chatbot's stated position.
- `--initial-p 2`: initial belief log-odds (about 88% probability of the scenario's
  false-belief proposition). Zero means uncertain; negative values reject it.
- `--condition low|medium|high`: the chatbot's sycophancy instructions, independent
  of the user's trust in it.
- `--scenario SCENARIO_ID`: selects the topic and initial belief proposition.

Trust is held fixed in each conversation. Belief strength evolves through the user
model's self-reports, not through the judge's score. These are simulated beliefs,
and trust prompts do not guarantee a particular ordering of observed belief drift.
The user bot is not given ground truth unless external conversational evidence is enabled
with, for example, `--evidence step`. Mathematical evidence is available only with
the legacy `--user-mode controlled` numerical user.

Trajectory CSVs include both agents' dialogue, pre/post-reply belief values, belief
change reasons, trust, and model metadata. The summary and plot include the update
after the final chatbot reply. `--user-trust all` writes separate CSVs and plots
for each trust level plus `closed_loop_summary.csv`.

Static sanity/pilot/full experiments retain their controlled message templates.
An LLM conversation uses one initial user call plus three calls per turn (chatbot,
judge, and user reaction). Seeds are passed to the user model, but provider
determinism is not guaranteed.

API requests allow four retries and a 90-second request timeout. Provider
`Retry-After` wait times are honored, including waits longer than 60 seconds.
Without that header, exponential backoff starts at 3 seconds and is capped at
60 seconds including jitter. SDK automatic retries stay disabled so the application
controls the retry count. Settings live in `sycophancy_testbed/config.py` and
apply to newly started processes.

## Configuring Models & Rate Limits

By default, the testbed operates with Groq models that provide sufficient daily request allocations:
- **`qwen/qwen3.8-27b`** (Default): 30 RPM, 1,000 RPD, 8,000 TPM, 200,000 TPD. Balanced reasoning and instruction following.
- **`openai/gpt-oss-120b`**: 30 RPM, 1,000 RPD, 8,000 TPM, 200,000 TPD. High-capacity reasoning model with internal thinking tokens.
- **`openai/gpt-oss-20b`**: 30 RPM, 1,000 RPD, 8,000 TPM, 200,000 TPD. Fast, lightweight reasoning model.

You can select models via CLI flags:
```bash
python -m sycophancy_testbed.main --run pilot --assistant-model qwen/qwen3.8-27b
```
Or directly from the **Target Model** dropdown in the Web UI.

> **Note on Rate Limits**: While these models provide 1,000 requests per day (RPD), running full closed-loop grids across all scenarios can rapidly consume quota. For large-scale automated testing, see `PC_Documentation/10_current_codebase_state_and_honest_audit.md` for local model hosting (Ollama / vLLM) strategies.

