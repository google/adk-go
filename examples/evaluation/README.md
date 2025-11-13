# Evaluation Framework Examples

This directory contains examples demonstrating the ADK evaluation framework for testing and measuring AI agent performance.

## Available Examples

### [Basic](./basic/) ⚡ **Start Here**
Simple introduction to LLM-based evaluation:
- Core evaluation setup
- 2 evaluators (algorithmic + LLM-as-Judge)
- Built-in rate limiting
- In-memory storage
- Clear result output

**Best for:** Getting started, understanding fundamentals

### [Comprehensive](./comprehensive/)
Production-ready example with all features:
- All 8 evaluation metrics
- Agent with custom tools
- File-based persistent storage
- Rubric-based evaluation
- Safety and hallucination detection
- Automatic rate limiting
- Detailed result reporting

**Best for:** Production use cases, advanced evaluation needs

## Quick Start

1. Set your API key:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

2. Try basic example (with LLM evaluation):
```bash
cd basic
go run main.go
```

3. Run comprehensive example (all features):
```bash
cd comprehensive
go run main.go
```

## ⚠️ Rate Limits

When using LLM-based evaluators, you may hit API rate limits. See [RATE_LIMITS.md](./RATE_LIMITS.md) for:
- How rate limiting works
- Configuration options
- Best practices
- Troubleshooting guide

**TL;DR:** The framework now includes automatic rate limiting.

## Evaluation Framework Overview

### Core Components

- **EvalSet**: Collection of test cases for systematic evaluation
- **EvalCase**: Single test scenario with conversation flow and expected outcomes
- **Evaluator**: Metric-specific evaluation logic
- **Runner**: Orchestrates evaluation execution
- **Storage**: Persists eval sets and results

### Available Metrics

#### Response Quality (4 metrics)
1. **RESPONSE_MATCH_SCORE** - ROUGE-1 algorithmic comparison
2. **SEMANTIC_RESPONSE_MATCH** - LLM-as-Judge semantic validation
3. **RESPONSE_EVALUATION_SCORE** - Coherence assessment (1-5 scale)
4. **RUBRIC_BASED_RESPONSE_QUALITY** - Custom quality criteria

#### Tool Usage (2 metrics)
5. **TOOL_TRAJECTORY_AVG_SCORE** - Exact tool sequence matching
6. **RUBRIC_BASED_TOOL_USE_QUALITY** - Custom tool quality criteria

#### Safety & Quality (2 metrics)
7. **SAFETY** - Harmlessness evaluation
8. **HALLUCINATIONS** - Unsupported claim detection

### Evaluation Methods

- **Algorithmic**: Fast, deterministic comparisons (ROUGE, exact matching)
- **LLM-as-Judge**: Flexible semantic evaluation with customizable rubrics
- **Multi-sample**: Increase reliability through multiple independent evaluations

## Use Cases

### Development Testing
```go
// Quick validation during development
config := &evaluation.EvalConfig{
    Criteria: map[string]evaluation.Criterion{
        "response_match": &evaluation.Threshold{MinScore: 0.7},
    },
}
```

### Production Quality Gates
```go
// Comprehensive quality checks before deployment
config := &evaluation.EvalConfig{
    Criteria: map[string]evaluation.Criterion{
        "response_quality": &evaluation.LLMAsJudgeCriterion{
            Threshold:  &evaluation.Threshold{MinScore: 0.8},
            MetricType: evaluation.MetricResponseQuality,
            NumSamples: 3,
        },
        "safety": &evaluation.LLMAsJudgeCriterion{
            Threshold:  &evaluation.Threshold{MinScore: 0.95},
            MetricType: evaluation.MetricSafety,
        },
    },
}
```

### Regression Testing
```go
// Track performance over time
evalStorage, _ := storage.NewFileStorage("./regression_tests")
// Run evals regularly and compare results
```

## Storage Options

### In-Memory
```go
evalStorage := storage.NewMemoryStorage()
```
- Fast, no persistence
- Ideal for testing and development

### File-Based
```go
evalStorage, err := storage.NewFileStorage("./eval_data")
```
- JSON persistence to disk
- Ideal for CI/CD and analysis

## Integration Patterns

### CI/CD Integration
Run evaluations in your pipeline:
```bash
go run ./evaluation_runner.go || exit 1
```

### REST API
Expose evaluation via HTTP endpoints (see comprehensive example)

### Custom Evaluators
Register your own domain-specific evaluators:
```go
evaluation.Register(myMetric, myEvaluatorFactory)
```

## Next Steps

1. Start with the **basic** example for LLM-based evaluation
2. Explore **comprehensive** for all features
3. Read [RATE_LIMITS.md](./RATE_LIMITS.md) for production configuration
4. Create custom eval sets for your use cases
5. Integrate into CI/CD

## Documentation

For detailed API documentation and usage guides, see:
- [evaluation/USAGE.md](../../evaluation/USAGE.md)
- [evaluation/doc.go](../../evaluation/doc.go)

## Requirements

- Go 1.24.4 or later
- Google API key (for Gemini models)
- ADK dependencies (automatically managed by Go modules)
