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

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/evaluation"
	"google.golang.org/adk/evaluation/evaluators"
	"google.golang.org/adk/evaluation/storage"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	model, err := gemini.NewModel(ctx, "gemini-2.0-flash-exp", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	agent, err := llmagent.New(llmagent.Config{
		Name:        "math_assistant",
		Model:       model,
		Description: "A helpful math assistant that answers basic math questions.",
		Instruction: "You are a math tutor. Answer math questions clearly and concisely.",
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	sessionService := session.InMemoryService()

	agentRunner, err := runner.New(runner.Config{
		AppName:        "math-eval-app",
		Agent:          agent,
		SessionService: sessionService,
	})
	if err != nil {
		log.Fatalf("Failed to create agent runner: %v", err)
	}

	if err := evaluation.RegisterDefaultEvaluators(map[evaluation.MetricType]evaluation.EvaluatorFactory{
		evaluation.MetricResponseMatch:         evaluators.NewResponseMatchEvaluator,
		evaluation.MetricSemanticResponseMatch: evaluators.NewSemanticResponseMatchEvaluator,
	}); err != nil {
		log.Fatalf("Failed to register evaluators: %v", err)
	}

	judgeLLM, err := gemini.NewModel(ctx, "gemini-2.0-flash-exp", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create judge LLM: %v", err)
	}

	evalStorage := storage.NewMemoryStorage()

	evalRunner := evaluation.NewRunner(evaluation.RunnerConfig{
		AgentRunner:    agentRunner,
		Storage:        evalStorage,
		SessionService: sessionService,
		AppName:        "math-eval-app",
		RateLimitDelay: 6 * time.Second,
	})

	evalSet := &evaluation.EvalSet{
		ID:   "basic-math-eval",
		Name: "Basic Math Evaluation",
		EvalCases: []evaluation.EvalCase{
			{
				ID: "addition-simple",
				Conversation: []evaluation.ConversationTurn{
					{Role: "user", Content: "What is 2 + 2?"},
				},
				ExpectedResponse: "2 + 2 = 4",
			},
			{
				ID: "multiplication-simple",
				Conversation: []evaluation.ConversationTurn{
					{Role: "user", Content: "What is 5 times 3?"},
				},
				ExpectedResponse: "5 times 3 = 15",
			},
		},
	}

	err = evalStorage.SaveEvalSet(ctx, "math-eval-app", evalSet)
	if err != nil {
		log.Fatalf("Failed to save eval set: %v", err)
	}

	config := &evaluation.EvalConfig{
		JudgeLLM:   judgeLLM,
		JudgeModel: "gemini-2.0-flash-exp",
		Criteria: []evaluation.Criterion{
			&evaluation.Threshold{
				MinScore:   0.5,
				MetricType: evaluation.MetricResponseMatch,
			},
			&evaluation.LLMAsJudgeCriterion{
				Threshold: &evaluation.Threshold{
					MinScore:   0.8,
					MetricType: evaluation.MetricSemanticResponseMatch,
				},
				MetricType: evaluation.MetricSemanticResponseMatch,
				JudgeModel: "gemini-2.0-flash-exp",
			},
		},
	}

	fmt.Println("Running evaluation...")
	fmt.Println("===================")

	result, err := evalRunner.RunEvalSet(ctx, evalSet, config)
	if err != nil {
		log.Fatalf("Evaluation failed: %v", err)
	}

	fmt.Printf("\nEvaluation Complete!\n")
	fmt.Printf("===================\n")
	fmt.Printf("Overall Status: %s\n", result.Status)
	fmt.Printf("Overall Score: %.2f\n\n", result.OverallScore)

	for i, caseResult := range result.EvalCaseResults {
		fmt.Printf("Case %d: %s\n", i+1, caseResult.EvalID)
		fmt.Printf("  Status: %s\n", caseResult.FinalEvalStatus)
		for metricName, metric := range caseResult.OverallMetricResults {
			fmt.Printf("  %s: %.2f (%s)\n", metricName, metric.Score, metric.Status)
			if metric.ErrorMessage != "" {
				fmt.Printf("    Error: %s\n", metric.ErrorMessage)
			}
		}
		fmt.Println()
	}

	fmt.Println("Evaluation results saved to storage.")
}
