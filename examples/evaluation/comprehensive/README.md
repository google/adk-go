# Comprehensive Evaluation Example

This example demonstrates all 8 evaluators of the ADK evaluation framework using a weather assistant agent with tool usage.

## Features Demonstrated

- Agent with custom tools (weather lookup)
- File-based storage for persistence
- All 8 evaluation metrics
- Tool trajectory evaluation
- Rubric-based evaluation
- Safety evaluation
- Hallucination detection
- Multi-sample LLM-as-Judge evaluation

## All 8 Evaluators Used

### Response Quality
1. **RESPONSE_MATCH_SCORE** - ROUGE-1 algorithmic comparison (0.0-1.0)
2. **SEMANTIC_RESPONSE_MATCH** - LLM-as-Judge semantic validation (0.0 or 1.0)
3. **RESPONSE_EVALUATION_SCORE** - Coherence assessment (1-5 scale)
4. **RUBRIC_BASED_RESPONSE_QUALITY** - Custom quality criteria (0.0-1.0)

### Tool Usage
5. **TOOL_TRAJECTORY_AVG_SCORE** - Exact tool sequence matching (0.0-1.0)
6. **RUBRIC_BASED_TOOL_USE_QUALITY** - Custom tool quality criteria (0.0-1.0)

### Safety & Quality
7. **SAFETY** - Harmlessness evaluation (0.0-1.0, higher = safer)
8. **HALLUCINATIONS** - Unsupported claim detection (0.0-1.0, higher = better)

## Running the Example

1. Set your API key:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

2. Run the example:
```bash
go run main.go
```

3. View persisted results:
```bash
ls -la eval_results/
```

## What to Expect

The example:
1. Creates a weather assistant with a custom tool
2. Sets up 3 evaluation cases:
   - Normal weather query (London)
   - Another weather query (Paris)
   - Harmful request (safety test)
3. Runs all 8 evaluators on each case
4. Displays comprehensive results with rubric breakdowns
5. Saves results to `./eval_results/` directory

## Sample Output

```
Comprehensive Evaluation Framework Demo
========================================
Registered Evaluators: 8
Eval Cases: 3
Criteria: 8

Running evaluation...

========================================
Evaluation Results
========================================
Overall Status: PASSED
Overall Score: 0.82
Completed: 2025-11-11 10:30:45

--- Case 1: weather-query-london ---
Status: PASSED

Metric Results:
  ✓ response_match: 0.75 (PASSED)
  ✓ semantic_match: 0.90 (PASSED)
  ✓ response_evaluation: 0.80 (PASSED)
  ✓ tool_trajectory: 1.00 (PASSED)
  ✓ tool_quality: 0.85 (PASSED)
      Rubric Scores:
        - accuracy: true
        - helpfulness: true
        - tool_usage: true
  ✓ response_quality: 0.88 (PASSED)
  ✓ safety: 0.95 (PASSED)
  ✓ hallucinations: 0.92 (PASSED)

Invocations: 1
  User: What's the weather like in London?
  Agent: The weather in London is currently 15°C and cloudy.
  Tools Used: 1
```

## Evaluation Storage

Results are persisted to `./eval_results/` in JSON format:
- `eval_sets/` - Stores evaluation test sets
- `eval_results/` - Stores evaluation run results

You can load and analyze previous results programmatically using the storage API.

## Customization

### Add Custom Rubrics

Modify the `Rubrics` field in eval cases:
```go
Rubrics: map[string]evaluation.Rubric{
    "custom_rubric": {
        RubricID:      "custom_rubric",
        RubricContent: "Your custom evaluation criteria",
    },
}
```

### Adjust Thresholds

Modify minimum scores in the config:
```go
"safety": &evaluation.LLMAsJudgeCriterion{
    Threshold: &evaluation.Threshold{
        MinScore: 0.95, // Stricter safety requirement
    },
    MetricType: evaluation.MetricSafety,
    JudgeModel: "gemini-2.0-flash-exp",
}
```

### Multi-Sample Evaluation

Increase reliability with multiple LLM samples:
```go
NumSamples: 5, // Run 5 independent evaluations and aggregate
```

## Next Steps

- Integrate with CI/CD for automated testing
- Create custom evaluators for domain-specific metrics
- Export results for analysis and reporting
- Use the REST API to run evaluations remotely
