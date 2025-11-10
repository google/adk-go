// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package evaluation provides a comprehensive framework for evaluating AI agent performance.
//
// The evaluation framework supports systematic testing of agent responses, tool usage,
// safety, and quality through a variety of metrics and evaluation methods.
//
// # Core Concepts
//
// EvalSet: A collection of test cases for systematic agent evaluation
//
// EvalCase: A single test scenario with conversation flow and expected outcomes
//
// Evaluator: Interface for metric-specific evaluation logic
//
// MetricResult: Detailed results including scores, rubric breakdowns, and judge responses
//
// # Supported Metrics
//
// The framework provides 8 comprehensive metrics:
//
// Response Quality Metrics:
//   - RESPONSE_MATCH_SCORE: ROUGE-1 algorithmic comparison (0.0-1.0)
//   - SEMANTIC_RESPONSE_MATCH: LLM-as-Judge semantic validation (0.0 or 1.0)
//   - RESPONSE_EVALUATION_SCORE: Coherence assessment (1-5 scale)
//   - RUBRIC_BASED_RESPONSE_QUALITY: Custom quality criteria (0.0-1.0)
//
// Tool Usage Metrics:
//   - TOOL_TRAJECTORY_AVG_SCORE: Exact sequence matching (0.0-1.0)
//   - RUBRIC_BASED_TOOL_USE_QUALITY: Custom tool quality criteria (0.0-1.0)
//
// Safety & Quality Metrics:
//   - SAFETY: Harmlessness evaluation (0.0-1.0, higher = safer)
//   - HALLUCINATIONS: Unsupported claim detection (0.0-1.0, higher = better)
//
// # LLM-as-Judge
//
// The framework includes a full LLM-as-Judge implementation with:
//   - Multi-sample evaluation with configurable sample count
//   - Response parsing (scores, verdicts, rubric responses)
//   - Aggregation logic (majority voting, averaging)
//   - Flexible prompt templates for different evaluation types
//
// # Storage Backends
//
// Multiple storage options are available:
//   - In-memory: Fast, suitable for testing and development
//   - File-based: JSON persistence for local storage
//
// # Example Usage
//
//	// Create an eval set
//	evalSet := &evaluation.EvalSet{
//	    ID:   "customer-service-eval",
//	    Name: "Customer Service Quality",
//	    EvalCases: []evaluation.EvalCase{
//	        {
//	            ID: "case-1",
//	            Conversation: []evaluation.ConversationTurn{
//	                {Role: "user", Content: "How do I reset my password?"},
//	            },
//	            ExpectedResponse: "You can reset your password by clicking...",
//	        },
//	    },
//	}
//
//	// Configure evaluation criteria
//	config := &evaluation.EvalConfig{
//	    Criteria: map[string]evaluation.Criterion{
//	        string(evaluation.MetricResponseMatch): &evaluation.Threshold{
//	            MinScore: 0.7,
//	        },
//	    },
//	}
//
//	// Run evaluation
//	runner := evaluation.NewRunner(agent, storage)
//	result, err := runner.RunEvalSet(ctx, evalSet, config)
//
// # Detailed Result Analysis
//
// Results include comprehensive tracking:
//   - Per-invocation breakdown (every user query + agent response)
//   - Tool trajectory details (actual vs expected tool calls)
//   - Rubric scores (individual rubric verdicts + explanations)
//   - Judge responses (raw LLM judge outputs)
//   - Session details (events, state, tokens, timing)
//
// # Registry Pattern
//
// Evaluators are registered in a global registry:
//
//	// Register a custom evaluator
//	evaluation.Register(customMetric, func(config EvaluatorConfig) (Evaluator, error) {
//	    return NewCustomEvaluator(config), nil
//	})
//
//	// Create evaluator from registry
//	evaluator, err := evaluation.CreateEvaluator(customMetric, config)
package evaluation
