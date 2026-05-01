# prompt_optimizer

Find the best prompt template for a small QA eval set. Demonstrates the
`optimize` + `eval` packages working together.

The setup:

- 6 candidate prompt templates (different framings of "answer the
  question concisely").
- A small eval set of 4 product-FAQ questions with reference answers.
- A stub agent that, for demonstration, simulates better answers when
  the prompt mentions "step-by-step" — this lets you see the search
  pick a winner deterministically.
- `optimize.GridSampler` walks every variant; `optimize.Search` scores
  each with `eval.ContainsScorer` against the eval set and ranks them
  best-first.

Replace the stub agent with a real LLM agent (calling your prompt
template) and the search becomes a real prompt-A/B harness.
