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

// Package provides a quickstart ADK agent.
package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"net/http"
	"os"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adkrest/controllers"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

func main() {
	log.SetOutput(os.Stdout)
	ctx := context.Background()

	// gemini-3.1-flash-live-preview
	// gemini-2.5-flash-native-audio-preview-12-2025
	model, err := gemini.NewModel(ctx, "gemini-3.1-flash-live-preview", &genai.ClientConfig{
		APIKey: os.Getenv("GOOGLE_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Define a streaming tool that yields numbers.
	counterTool, err := functiontool.NewStreaming(functiontool.Config{
		Name:        "count_to",
		Description: "Counts to a specified number, yielding each number with a delay.",
	}, func(ctx tool.Context, args struct {
		N int `json:"n"`
	},
	) iter.Seq2[string, error] {
		return func(yield func(string, error) bool) {
			for i := 1; i <= args.N; i++ {
				time.Sleep(5000 * time.Millisecond)
				if !yield(fmt.Sprintf("Count: %d", i), nil) {
					return
				}
			}
		}
	})
	if err != nil {
		log.Fatalf("Failed to create streaming tool: %v", err)
	}

	// Define a standard tool to stop streaming (mock).
	stopTool, err := functiontool.New(functiontool.Config{
		Name:        "stop_streaming",
		Description: "Stops a running streaming function.",
	}, func(ctx tool.Context, args struct {
		FunctionName string `json:"function_name"`
	},
	) (map[string]any, error) {
		return map[string]any{"status": fmt.Sprintf("Requested to stop %s", args.FunctionName)}, nil
	})
	if err != nil {
		log.Fatalf("Failed to create stop tool: %v", err)
	}

	// Define a function tool to check divisibility.
	checkDivisibleTool, err := functiontool.New(functiontool.Config{
		Name:        "check_divisible",
		Description: "Checks if a number is divisible by another number.",
	}, func(ctx tool.Context, args struct {
		Number  int `json:"number"`
		Divisor int `json:"divisor"`
	},
	) (map[string]any, error) {
		if args.Divisor == 0 {
			return map[string]any{"result": false, "error": "cannot divide by zero"}, nil
		}
		fmt.Printf("Dividing %d by %d\n", args.Number, args.Divisor)
		return map[string]any{"result": args.Number%args.Divisor == 0}, nil
	})
	if err != nil {
		log.Fatalf("Failed to create check divisible tool: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "bidi-demo",
		Model:       model,
		Instruction: "You are a helpful assistant with a streaming tool 'count_to'. Always use it when asked to count. Wait for the tool results, and when you recieve them you should say the number, if it is divisible by 3 you should not say the number and instead say Fizz and if it is divisible by 5 you should say Buzz, if it is divisible by both 3 and 5 you should say FizzBuzz. Always use the check_divisible tool",
		Tools:       []tool.Tool{counterTool, stopTool, checkDivisibleTool},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	uiMode := true

	if uiMode {
		// Create runner
		ss := session.InMemoryService()

		fs := http.FileServer(http.Dir("examples/bidi/static"))
		http.Handle("/", fs)
		http.Handle("/static/", http.StripPrefix("/static/", fs))

		controller := controllers.NewRuntimeAPIController(ss, nil, agent.NewSingleLoader(a), nil, 0, runner.PluginConfig{}, true)

		http.HandleFunc("/run_live", func(w http.ResponseWriter, req *http.Request) {
			err := controller.RunLiveHandler(w, req)
			if err != nil {
				log.Printf("RunLiveHandler failed: %v", err)
			}
		})

		fmt.Println("Serving UI on http://localhost:8081")
		log.Fatal(http.ListenAndServe(":8081", nil))
	}
}
