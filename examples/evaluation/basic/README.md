# Basic Evaluation Example

This example demonstrates the core functionality of the ADK evaluation framework with a simple math assistant agent.

## Features Demonstrated

- Creating a simple agent for evaluation
- Setting up evaluation storage (in-memory)
- Registering evaluators
- Creating an eval set with test cases
- Configuring evaluation criteria
- Running evaluations and viewing results

## Evaluators Used

1. **RESPONSE_MATCH_SCORE** - Algorithmic comparison using ROUGE-1
2. **SEMANTIC_RESPONSE_MATCH** - LLM-as-Judge semantic validation

## Running the Example

1. Set your API key:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

2. Run the example:
```bash
go run main.go
```

## What to Expect

The example:
1. Creates a math assistant agent
2. Sets up two evaluation cases (addition and multiplication)
3. Runs both evaluators on each case
4. Displays detailed results including scores and pass/fail status

## Sample Output

```
Running evaluation...
===================

Evaluation Complete!
===================
Overall Status: PASSED
Overall Score: 0.85

Case 1: addition-simple
  Status: PASSED
  response_match: 0.82 (PASSED)
  semantic_match: 0.90 (PASSED)

Case 2: multiplication-simple
  Status: PASSED
  response_match: 0.78 (PASSED)
  semantic_match: 0.88 (PASSED)

Evaluation results saved to storage.
```
