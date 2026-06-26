# Evaluation Framework Usage Guide

## Quick Start

Here's how to set up the evaluation framework:

### 1. Register Evaluators

First, register all evaluators with the registry:

```go
import (
    "google.golang.org/adk/evaluation"
    "google.golang.org/adk/evaluation/evaluators"
)

func init() {
    // Register all 8 built-in evaluators
    evaluation.RegisterDefaultEvaluators(map[evaluation.MetricType]evaluation.EvaluatorFactory{
        evaluation.MetricResponseMatch:          evaluators.NewResponseMatchEvaluator,
        evaluation.MetricSemanticResponseMatch:  evaluators.NewSemanticResponseMatchEvaluator,
        evaluation.MetricResponseEvaluationScore: evaluators.NewResponseEvaluationScoreEvaluator,
        evaluation.MetricToolTrajectoryAvgScore:  evaluators.NewToolTrajectoryEvaluator,
        evaluation.MetricToolUseQuality:          evaluators.NewToolUseQualityEvaluator,
        evaluation.MetricResponseQuality:         evaluators.NewResponseQualityEvaluator,
        evaluation.MetricSafety:                  evaluators.NewSafetyEvaluator,
        evaluation.MetricHallucinations:          evaluators.NewHallucinationsEvaluator,
    })
}
```

### 2. Create LLM for Evaluation

Create an LLM instance for the judge (used by LLM-as-Judge evaluators):

```go
import (
    "context"
    "google.golang.org/adk/model/gemini"
    "google.golang.org/genai"
)

ctx := context.Background()
judgeLLM, err := gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
    APIKey: os.Getenv("GOOGLE_API_KEY"),
})
if err != nil {
    log.Fatal(err)
}
```

### 3. Create Storage Backend

Choose a storage backend for eval sets and results:

```go
import "google.golang.org/adk/evaluation/storage"

// Option A: In-memory (for testing)
evalStorage := storage.NewMemoryStorage()

// Option B: File-based (for persistence)
evalStorage, err := storage.NewFileStorage("./eval_data")
if err != nil {
    log.Fatal(err)
}
```

### 4. Create Evaluation Runner

Create a runner with your agent and configuration:

```go
import (
    "google.golang.org/adk/evaluation"
    "google.golang.org/adk/runner"
)

// Your existing agent runner
agentRunner := runner.NewRunner(/* your agent config */)

// Create evaluation runner
evalRunner := evaluation.NewRunner(evaluation.RunnerConfig{
    AgentRunner: agentRunner,
    Storage:     evalStorage,
    JudgeLLM:    judgeLLM,
    JudgeModel:  "gemini-2.5-flash",
})
```

### 5. Set Up HTTP API (Optional)

If you want to expose evaluation via REST API:

```go
import (
    "google.golang.org/adk/server/restapi/handlers"
    "google.golang.org/adk/server/restapi/routers"
)

// Create evaluation handler
evalHandler := handlers.NewEvalHandler(evalStorage, evalRunner)

// Create router with handler
evalRouter := routers.NewEvalAPIRouter(evalHandler)

// Add to your server
// (router setup depends on your server configuration)
```

## Using the Evaluation Framework

### Create an Evaluation Set

```go
evalSet := &evaluation.EvalSet{
    ID:   "customer-service",
    Name: "Customer Service Quality Evaluation",
    EvalCases: []evaluation.EvalCase{
        {
            ID: "case-1",
            Conversation: []evaluation.ConversationTurn{
                {Role: "user", Content: "How do I reset my password?"},
            },
            ExpectedResponse: "You can reset your password by clicking the 'Forgot Password' link...",
            Rubrics: map[string]evaluation.Rubric{
                "helpful": {
                    RubricID:      "helpful",
                    RubricContent: "Response is helpful and actionable",
                },
                "concise": {
                    RubricID:      "concise",
                    RubricContent: "Response is concise and to the point",
                },
            },
        },
    },
}

// Save eval set
err := evalStorage.SaveEvalSet(ctx, "my-app", evalSet)
```

### Define Evaluation Criteria

```go
config := &evaluation.EvalConfig{
    Criteria: map[string]evaluation.Criterion{
        // Algorithmic comparison (no LLM needed)
        "response_match": &evaluation.Threshold{
            MinScore: 0.7,
        },

        // LLM-as-Judge with rubrics
        "response_quality": &evaluation.LLMAsJudgeCriterion{
            Threshold: &evaluation.Threshold{
                MinScore: 0.8,
            },
            MetricType:  evaluation.MetricResponseQuality,
            JudgeModel:  "gemini-2.5-flash",
            NumSamples:  3, // Use 3 samples for reliability
        },

        // Safety evaluation
        "safety": &evaluation.LLMAsJudgeCriterion{
            Threshold: &evaluation.Threshold{
                MinScore: 0.9, // Must be very safe
            },
            MetricType: evaluation.MetricSafety,
            JudgeModel: "gemini-2.5-flash",
        },
    },
}
```

### Run Evaluation

```go
result, err := evalRunner.RunEvalSet(ctx, evalSet, config)
if err != nil {
    log.Fatal(err)
}

// Check results
fmt.Printf("Overall Score: %.2f\n", result.OverallScore)
fmt.Printf("Status: %s\n", result.Status)

for _, caseResult := range result.EvalCaseResults {
    fmt.Printf("\nCase %s: %s\n", caseResult.EvalID, caseResult.FinalEvalStatus)
    for metricName, metric := range caseResult.OverallMetricResults {
        fmt.Printf("  %s: %.2f (%s)\n", metricName, metric.Score, metric.Status)
    }
}
```

## REST API Endpoints

Once configured, the evaluation API exposes these endpoints:

- `GET /apps/{app_name}/eval_sets` - List evaluation sets
- `POST /apps/{app_name}/eval_sets` - Create evaluation set
- `GET /apps/{app_name}/eval_sets/{eval_set_name}` - Get specific set
- `POST /apps/{app_name}/eval_sets/{eval_set_name}` - Run evaluation
- `DELETE /apps/{app_name}/eval_sets/{eval_set_name}` - Delete set
- `GET /apps/{app_name}/eval_results` - List results
- `GET /apps/{app_name}/eval_results/{result_id}` - Get specific result

## All 8 Metrics Available

1. **RESPONSE_MATCH_SCORE** - ROUGE-1 algorithmic comparison (no LLM)
2. **SEMANTIC_RESPONSE_MATCH** - LLM-as-Judge semantic validation
3. **RESPONSE_EVALUATION_SCORE** - Coherence assessment (1-5 scale)
4. **TOOL_TRAJECTORY_AVG_SCORE** - Exact tool sequence matching (no LLM)
5. **RUBRIC_BASED_TOOL_USE_QUALITY** - Custom tool quality criteria (LLM)
6. **RUBRIC_BASED_RESPONSE_QUALITY** - Custom response quality (LLM)
7. **SAFETY** - Harmlessness evaluation (LLM)
8. **HALLUCINATIONS** - Unsupported claim detection (2-step LLM process)

## Features

- **Flexible Architecture** - Pluggable evaluators, storage backends, and LLM providers
- **Detailed Results** - Per-invocation tracking, rubric scores, and judge responses
- **REST API** - HTTP interface for remote evaluation
- **Type Safe** - Strong Go typing with proper interfaces
- **Thread Safe** - Concurrent evaluation support with safe storage operations

## Integration Workflow

1. Register evaluators in your main.go or init function
2. Create evaluation sets with test cases
3. Configure evaluation criteria with desired metrics
4. Run evaluations and analyze results
5. Optionally expose evaluation via HTTP API
