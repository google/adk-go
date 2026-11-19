// Copyright 2026 Google LLC
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

package workflow_test

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// printOutputs runs w once for message and prints the path and output of
// every node event that carries an output, in the order the run yields them.
// A path is "<workflow>@<run id>/<node>@<run id>".
func printOutputs(w *workflow.Workflow, message string) error {
	ctx := context.Background()
	a, err := agent.New(agent.Config{Name: w.Name(), Run: w.Run})
	if err != nil {
		return err
	}
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		return err
	}
	msg := genai.NewContentFromText(message, genai.RoleUser)
	for ev, err := range r.Run(ctx, "user", "session", msg, agent.RunConfig{}) {
		if err != nil {
			return err
		}
		if ev.NodeInfo != nil && ev.Output != nil {
			fmt.Printf("%s: %v\n", ev.NodeInfo.Path, ev.Output)
		}
	}
	return nil
}

// Chain wires nodes into a sequence that starts at Start. Each node's
// output is the next node's input, converted to the type the next node's
// function takes; the first node receives the user's message.
func ExampleChain() {
	upper := workflow.NewFunctionNode("upper", func(_ agent.Context, in string) (string, error) {
		return strings.ToUpper(in), nil
	}, workflow.NodeConfig{})
	count := workflow.NewFunctionNode("count", func(_ agent.Context, in string) (int, error) {
		return len(in), nil
	}, workflow.NodeConfig{})
	report := workflow.NewFunctionNode("report", func(_ agent.Context, n int) (string, error) {
		return fmt.Sprintf("%d letters", n), nil
	}, workflow.NodeConfig{})

	w, err := workflow.New("pipeline", workflow.Chain(workflow.Start, upper, count, report))
	if err != nil {
		log.Fatal(err)
	}
	if err := printOutputs(w, "hello"); err != nil {
		log.Fatal(err)
	}
	// Output:
	// pipeline@1/upper@1: HELLO
	// pipeline@1/count@1: 5
	// pipeline@1/report@1: 5 letters
}

// A node with several outgoing edges fans out: every successor runs with
// the node's output. A JoinNode fans the branches back in. It runs once every
// node with an edge into it has finished, and its input and output are a
// map from each predecessor's name to that predecessor's output.
func ExampleNewJoinNode() {
	split := workflow.NewFunctionNode("split", func(_ agent.Context, in string) (string, error) {
		return in, nil
	}, workflow.NodeConfig{})
	words := workflow.NewFunctionNode("words", func(_ agent.Context, in string) (int, error) {
		return len(strings.Fields(in)), nil
	}, workflow.NodeConfig{})
	chars := workflow.NewFunctionNode("chars", func(_ agent.Context, in string) (int, error) {
		return len(in), nil
	}, workflow.NodeConfig{})
	join := workflow.NewJoinNode("join")
	summary := workflow.NewFunctionNode("summary", func(_ agent.Context, in map[string]any) (string, error) {
		return fmt.Sprintf("%v words, %v characters", in["words"], in["chars"]), nil
	}, workflow.NodeConfig{})

	edges := workflow.NewEdgeBuilder().
		Add(workflow.Start, split).
		Add(split, words).
		Add(split, chars).
		AddFanIn(join, words, chars).
		Add(join, summary).
		Build()
	w, err := workflow.New("stats", edges)
	if err != nil {
		log.Fatal(err)
	}

	// Print only the join and what follows it: words and chars run
	// concurrently, so the order of their outputs differs between runs.
	a, err := agent.New(agent.Config{Name: w.Name(), Run: w.Run})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("to be or not", genai.RoleUser)
	for ev, err := range r.Run(context.Background(), "user", "session", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatal(err)
		}
		if ev.NodeInfo == nil || ev.Output == nil {
			continue
		}
		switch ev.NodeInfo.Path {
		case "stats@1/join@1", "stats@1/summary@1":
			fmt.Printf("%s: %v\n", ev.NodeInfo.Path, ev.Output)
		}
	}
	// Output:
	// stats@1/join@1: map[chars:12 words:4]
	// stats@1/summary@1: 4 words, 12 characters
}

// AddRoutes adds one edge per entry of a routing map, each with a
// StringRoute of the entry's key. A node selects edges by the strings in the
// Routes of the event it returns, and an edge with the Default route runs
// when no other routed edge of the node matched.
func ExampleEdgeBuilder_AddRoutes() {
	classify := workflow.NewFunctionNode("classify", func(ctx agent.Context, in string) (*session.Event, error) {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Output = in
		switch {
		case strings.HasSuffix(in, "?"):
			ev.Routes = []string{"question"}
		case strings.HasSuffix(in, "!"):
			ev.Routes = []string{"exclamation"}
		default:
			ev.Routes = []string{"statement"}
		}
		return ev, nil
	}, workflow.NodeConfig{})
	reply := func(name, prefix string) workflow.Node {
		return workflow.NewFunctionNode(name, func(_ agent.Context, in string) (string, error) {
			return prefix + in, nil
		}, workflow.NodeConfig{})
	}

	b := workflow.NewEdgeBuilder().
		Add(workflow.Start, classify).
		AddRoutes(classify, map[string]workflow.Node{
			"question":    reply("answer", "answering: "),
			"exclamation": reply("calm", "calm down: "),
		}).
		AddRoute(classify, reply("note", "noted: "), workflow.Default)
	if err := b.Err(); err != nil {
		log.Fatal(err)
	}
	w, err := workflow.New("router", b.Build())
	if err != nil {
		log.Fatal(err)
	}
	for _, msg := range []string{"Is it late?", "It is late!", "It is late."} {
		if err := printOutputs(w, msg); err != nil {
			log.Fatal(err)
		}
	}
	// Output:
	// router@1/classify@1: Is it late?
	// router@1/answer@1: answering: Is it late?
	// router@1/classify@1: It is late!
	// router@1/calm@1: calm down: It is late!
	// router@1/classify@1: It is late.
	// router@1/note@1: noted: It is late.
}

// A graph node with NodeConfig.RetryConfig runs again after a failed attempt
// while attempts remain. The failed attempt's error is not the run's error;
// once MaxAttempts attempts have failed, the last error ends the run, wrapped
// so that errors.Is finds the node's own error.
func ExampleRetryConfig() {
	errUnavailable := errors.New("service unavailable")
	flaky := func(failures int) workflow.Node {
		return workflow.NewFunctionNode("fetch", func(ctx agent.Context, _ string) (string, error) {
			n := agent.AttemptCount(ctx)
			fmt.Println("attempt", n)
			if n <= failures {
				return "", errUnavailable
			}
			return "fetched", nil
		}, workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{MaxAttempts: 3}})
	}

	recovers, err := workflow.New("recovers", workflow.Chain(workflow.Start, flaky(2)))
	if err != nil {
		log.Fatal(err)
	}
	if err := printOutputs(recovers, "go"); err != nil {
		log.Fatal(err)
	}

	givesUp, err := workflow.New("gives_up", workflow.Chain(workflow.Start, flaky(3)))
	if err != nil {
		log.Fatal(err)
	}
	err = printOutputs(givesUp, "go")
	fmt.Println("run error is errUnavailable:", errors.Is(err, errUnavailable))
	// Output:
	// attempt 1
	// attempt 2
	// attempt 3
	// recovers@1/fetch@1: fetched
	// attempt 1
	// attempt 2
	// attempt 3
	// run error is errUnavailable: true
}
