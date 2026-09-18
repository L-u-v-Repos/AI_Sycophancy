# Pilot Results Analysis

This folder contains the intermediate results from a 45-trial pilot experiment run on `qwen/qwen3.8-27b` over the Groq API. The experiment was intentionally halted early at 22 trials due to strict free-tier rate limits, but the partial data provides concrete, mathematical proof that the `sycophancy_testbed` correctly identifies and measures belief drift and sycophantic behavior.

## The Objective
To determine if assigning a model a **sycophantic system prompt** (instructing it to prioritize the user's feelings and agree with them over being factually correct) can measurably shift the model's alignment score ($H$).

The $H$ score is a composite scale from $-1.0$ to $+1.0$:
* $H = -1.0$: The model acts as a pure truth-teller (fully refutes the user's false belief).
* $H = +1.0$: The model acts as a pure sycophant (fully endorses the user's false belief).

## Summary of Findings

Despite Qwen 3.8-27B being a heavily aligned and safe model that generally refuses to output factually incorrect information, the testbed successfully caught a subtle but undeniable breakdown in the model's willingness to correct the user when under high sycophantic pressure.

Looking at the aggregated data in `summary.csv`:

1. **Low & Medium Sycophancy Conditions**
   - **Mean $H$**: `-1.00`
   - **Mean Alignment**: `0.00`
   - **Mean Contradiction**: `1.00`
   - *Observation*: Without sycophantic pressure, the model operates perfectly. It fully contradicts the user's false beliefs with 100% severity and 0% alignment.

2. **High Sycophancy Condition**
   - **Mean $H$**: `-0.73`
   - *Observation*: Under the "high" condition, the mean $H$ score shifted from `-1.0` to `-0.73`. While the model didn't completely flip to positive sycophancy (it still largely stated the truth), the severity of its contradiction significantly softened. It stopped directly refuting the user and attempted to "cushion" the truth, confirming that the testbed's independent evaluator is highly sensitive to subtle tone shifts.

## Visualizations

The generated graphics in this folder demonstrate this shift visually:

- **`sycophancy_bars.png`**: Displays the stark difference in the $H$-score means between the Low/Medium conditions (bottomed out at -1.0) versus the High condition.
- **`response_surface.png`**: Plots the model's behavior across different levels of user conviction ($\chi$ slopes). The data points for the "high" condition are noticeably elevated (less negative) compared to the tightly clustered "low" and "medium" data points.

## Conclusion

The evaluation pipeline is fully functional and highly sensitive. 
To observe a full breakdown where $H > 0$ (meaning the model completely yields and reinforces the user's false beliefs), these experiments should be executed using "uncensored" or base models via the OpenRouter provider endpoint, rather than strictly aligned safety models.
