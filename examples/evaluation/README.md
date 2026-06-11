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
- All 8 evaluation metrics
- Agent with custom tools
- File-based persistent storage
- Rubric-based evaluation
- Safety and hallucination detection
- Automatic rate limiting
- Detailed result reporting

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
## Evaluation Framework Overview

### Core Components

- **EvalSet**: Collection of test cases for systematic evaluation
- **EvalCase**: Single test scenario with conversation flow and expected outcomes
- **Evaluator**: Metric-specific evaluation logic
- **Runner**: Orchestrates evaluation execution
- **Storage**: Persists eval sets and results

### Available Metrics

#### Response Quality
1. **RESPONSE_MATCH_SCORE** - ROUGE-1 algorithmic comparison
2. **SEMANTIC_RESPONSE_MATCH** - LLM-as-Judge semantic validation
3. **RESPONSE_EVALUATION_SCORE** - Coherence assessment (1-5 scale)
4. **RUBRIC_BASED_RESPONSE_QUALITY** - Custom quality criteria

#### Tool Usage
5. **TOOL_TRAJECTORY_AVG_SCORE** - Exact tool sequence matching
6. **RUBRIC_BASED_TOOL_USE_QUALITY** - Custom tool quality criteria

#### Safety & Quality
7. **SAFETY** - Harmlessness evaluation
8. **HALLUCINATIONS** - Unsupported claim detection

### Evaluation Methods

- **Algorithmic**: Fast, deterministic comparisons (ROUGE, exact matching)
- **LLM-as-Judge**: Flexible semantic evaluation with customizable rubrics

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
## Requirements

- Go 1.24.4 or later
- Google API key (for Gemini models)
- ADK dependencies (automatically managed by Go modules)
