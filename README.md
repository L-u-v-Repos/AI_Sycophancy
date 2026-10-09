# AI Sycophancy Testbed

A production-grade Python testbed designed to measure **sycophancy** and **belief drift** in Large Language Models (LLMs). This framework runs both static (single-turn) and closed-loop (multi-turn) experiments to evaluate how assistant models react to users holding false beliefs under different sycophantic pressure conditions.

## Features
- **Configurable System Prompts**: Evaluates models under Low, Medium, and High sycophancy instructions.
- **Robust API Client**: Built-in exponential backoff, jitter, and automatic retry handling for Groq and OpenRouter APIs.
- **Impartial Evaluator**: Uses an LLM judge to extract continuous scales of alignment and contradiction from Assistant responses.
- **Closed-Loop Dynamics**: Simulates an independent User Agent whose belief state drifts over multiple turns based on Assistant feedback and external evidence schedules.
- **Automated Visualizations**: Generates sycophancy metric bars and response surface susceptibility slopes.

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
   Open `.env` and set three separate keys: `ASSISTANT_API_KEY` (chatbot),
   `USER_API_KEY` (user bot), and `EVALUATOR_API_KEY` (judge).
   The default provider is Groq. The chatbot and judge use `--provider`.

## Usage

Run the `main.py` entrypoint to execute experiments. Results (CSV dumps and generated graphs) are automatically organized into timestamped folders inside `results/`.

### Run a Sanity Check
A quick 2-trial run to ensure the pipeline and APIs are functioning:
```bash
python -m sycophancy_testbed.main --run sanity
```

### Run a Pilot Experiment
Run a 45-trial experiment over a small subset of scenarios:
```bash
python -m sycophancy_testbed.main --run pilot --provider groq
```

### Run a Full Static Experiment
Run the full testbed across all scenarios and user conviction grids:
```bash
python -m sycophancy_testbed.main --run full
```

### Run a Closed-Loop Multi-Turn Simulation
The user is now an LLM bot by default. It starts with the selected scenario's belief
and generates its own messages and belief updates after each chatbot reply.
Compare three independent conversations with low, medium, and high trust in the chatbot:
```bash
python -m sycophancy_testbed.main --run closed_loop --turns 6 --user-trust all --initial-p 2 --condition high
```

Set `ASSISTANT_API_KEY`, `USER_API_KEY`, and `EVALUATOR_API_KEY` in `.env`.
Each role uses its own credential; missing keys cause an error before API calls.
Static runs need only the assistant and evaluator keys.
Use `--user-provider groq` or `--user-provider openrouter` and `--user-model MODEL_ID`
to select its provider and model. By default, it uses the assistant's provider and
model. The judge uses the assistant's provider and `--evaluator-model`.
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

## Configuring Models
By default, the testbed uses Groq. You can customize the default models in `sycophancy_testbed/config.py` or override them directly via CLI flags:
```bash
python -m sycophancy_testbed.main --run pilot --assistant-model qwen/qwen3.8-27b
```

*(Note: Extremely aligned models like `openai/gpt-oss-120b` will consistently yield an H-score of -1.0, showing extreme robustness. If you wish to observe model breakdown and induced sycophancy (H > 0), use smaller, less-aligned, or "uncensored" models via OpenRouter).*
