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
   Open `.env` and set your `GROQ_API_KEY` (or `OPENROUTER_API_KEY`).

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
Simulate a continuous conversation where the user updates their belief state based on the assistant's sycophancy:
```bash
python -m sycophancy_testbed.main --run closed_loop --turns 6 --evidence step
```

## Configuring Models
By default, the testbed uses Groq. You can customize the default models in `sycophancy_testbed/config.py` or override them directly via CLI flags:
```bash
python -m sycophancy_testbed.main --run pilot --assistant-model qwen/qwen3.8-27b
```

*(Note: Extremely aligned models like `openai/gpt-oss-120b` will consistently yield an H-score of -1.0, showing extreme robustness. If you wish to observe model breakdown and induced sycophancy (H > 0), use smaller, less-aligned, or "uncensored" models via OpenRouter).*
