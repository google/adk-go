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

// Command complex demonstrates a fan-out / fan-in research pipeline built
// on the graph workflow engine: three researcher agents run concurrently,
// a JoinNode barrier gathers their findings, a function node formats them,
// and a single-turn synthesis agent merges everything into one structured
// report.
//
//	START ─┬─> RenewableEnergyResearcher ─┐
//	       ├─> EVResearcher ──────────────┼─> gather ─> format ─> SynthesisAgent
//	       └─> CarbonCaptureResearcher ───┘   (Join)   (func)      (LLM)
//
// What it shows:
//   - fan-out: three AgentNodes wired straight from Start run in parallel;
//   - fan-in: a JoinNode waits for every researcher and hands the next node
//     a map[nodeName]output;
//   - a FunctionNode transforming the join's map into a single prompt;
//   - a single-turn AgentNode consuming a predecessor's output mid-graph;
//   - per-node RetryConfig guarding the flaky (model + search) calls;
//   - the default in-memory session service (nothing to configure).
//
// Requires GOOGLE_API_KEY in the environment.
//
//	export GOOGLE_API_KEY=...
//	go run ./examples/workflow/complex/ console
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/geminitool"
	"google.golang.org/adk/workflow"
)

// modelName is the Gemini model every agent in the pipeline uses. Google
// Search grounding (geminitool.GoogleSearch) requires a Gemini 2 model.
const modelName = "gemini-flash-latest"

// Agent names double as workflow node names. The JoinNode keys its output
// map by predecessor node name, so the formatter below looks results up by
// these same constants.
const (
	renewableResearcher       = "RenewableEnergyResearcher"
	electricVehicleResearcher = "EVResearcher"
	carbonResearcher          = "CarbonCaptureResearcher"
	synthesisAgent            = "SynthesisAgent"
)

// Each researcher targets one fixed, independent topic — the whole point of
// fanning out is that no researcher depends on another's result.
const (
	renewableInstruction = `You are an AI research assistant specializing in energy.
Research the latest advancements in renewable energy sources using the Google Search tool.
Summarize your key findings concisely in 1-2 sentences. Output only the summary.`

	electricVehicleInstruction = `You are an AI research assistant specializing in transportation.
Research the latest developments in electric vehicle technology using the Google Search tool.
Summarize your key findings concisely in 1-2 sentences. Output only the summary.`

	carbonInstruction = `You are an AI research assistant specializing in climate solutions.
Research the current state of carbon capture methods using the Google Search tool.
Summarize your key findings concisely in 1-2 sentences. Output only the summary.`

	synthesisInstruction = `You are an AI assistant that combines research summaries into one structured report.

Synthesize the "Input Summaries" you are given into a coherent report, attributing each
finding to its topic. Ground your report exclusively on the provided summaries; do not add
outside facts.

Output format:

## Recent Sustainable Technology Advancements

### Renewable Energy
[elaborate only on the renewable energy summary]

### Electric Vehicles
[elaborate only on the electric vehicle summary]

### Carbon Capture
[elaborate only on the carbon capture summary]

### Overall Conclusion
[1-2 sentences connecting only the findings above]`
)

// formatSummaries turns the JoinNode's map[nodeName]output into the single
// "Input Summaries" prompt consumed by the synthesis agent. Iterating a
// fixed slice (rather than ranging the map) keeps the prompt deterministic.
func formatSummaries(_ agent.Context, gathered map[string]any) (string, error) {
	sections := []struct{ label, key string }{
		{"Renewable Energy", renewableResearcher},
		{"Electric Vehicles", electricVehicleResearcher},
		{"Carbon Capture", carbonResearcher},
	}

	var sb strings.Builder
	sb.WriteString("Input Summaries:\n\n")
	for _, s := range sections {
		summary := "(no findings)"
		if v, ok := gathered[s.key]; ok && v != nil {
			if text := strings.TrimSpace(fmt.Sprint(v)); text != "" {
				summary = text
			}
		}
		fmt.Fprintf(&sb, "## %s\n%s\n\n", s.label, summary)
	}
	return sb.String(), nil
}

// newResearchPipeline builds the workflow agent. Splitting it out of main
// keeps main thin and the graph wiring easy to follow; nothing here calls the
// model, it only assembles the graph.
func newResearchPipeline(m model.LLM) (agent.Agent, error) {
	researcher := func(name, instruction string) (agent.Agent, error) {
		return llmagent.New(llmagent.Config{
			Name:        name,
			Model:       m,
			Description: "researches one topic and returns a short summary",
			Instruction: instruction,
			// Each researcher grounds its summary with Google Search.
			// The tool runs inside the Gemini model; no local execution.
			Tools: []tool.Tool{geminitool.GoogleSearch{}},
		})
	}

	renewableAgent, err := researcher(renewableResearcher, renewableInstruction)
	if err != nil {
		return nil, fmt.Errorf("creating renewable researcher: %w", err)
	}
	electricVehicleAgent, err := researcher(electricVehicleResearcher, electricVehicleInstruction)
	if err != nil {
		return nil, fmt.Errorf("creating EV researcher: %w", err)
	}
	carbonAgent, err := researcher(carbonResearcher, carbonInstruction)
	if err != nil {
		return nil, fmt.Errorf("creating carbon researcher: %w", err)
	}

	synthAgent, err := llmagent.New(llmagent.Config{
		Name:        synthesisAgent,
		Model:       m,
		Description: "merges the research summaries into one structured report",
		Instruction: synthesisInstruction,
	})
	if err != nil {
		return nil, fmt.Errorf("creating synthesis agent: %w", err)
	}

	// Retry the flaky model + search calls with exponential backoff. The
	// deterministic formatter needs no retries.
	llmNodeConfig := workflow.NodeConfig{
		RetryConfig: &workflow.RetryConfig{
			MaxAttempts:   3,
			InitialDelay:  time.Second,
			MaxDelay:      10 * time.Second,
			BackoffFactor: 2.0,
			Jitter:        0.5,
		},
	}

	// Wrapping an LlmAgent in an AgentNode defaults it to single-turn mode,
	// so the synthesis node consumes the formatter's output instead of the
	// chat history — which is why it may sit mid-graph rather than at Start.
	renewableNode, err := workflow.NewAgentNode(renewableAgent, llmNodeConfig)
	if err != nil {
		return nil, fmt.Errorf("creating renewable node: %w", err)
	}
	electricVehicleNode, err := workflow.NewAgentNode(electricVehicleAgent, llmNodeConfig)
	if err != nil {
		return nil, fmt.Errorf("creating EV node: %w", err)
	}
	carbonNode, err := workflow.NewAgentNode(carbonAgent, llmNodeConfig)
	if err != nil {
		return nil, fmt.Errorf("creating carbon node: %w", err)
	}
	synthNode, err := workflow.NewAgentNode(synthAgent, llmNodeConfig)
	if err != nil {
		return nil, fmt.Errorf("creating synthesis node: %w", err)
	}

	// gather is the fan-in barrier: it fires once, after every researcher
	// has finished, exposing a map[nodeName]output to its successor.
	gatherNode := workflow.NewJoinNode("gather")
	formatNode := workflow.NewFunctionNode("format_summaries", formatSummaries, workflow.NodeConfig{})

	// Graph wiring:
	//
	//   START ─┬─> renewable      ──┐
	//          ├─> electricVehicle──┼─> gather ─> format ─> synthesis
	//          └─> carbon         ──┘
	eb := workflow.NewEdgeBuilder()
	eb.AddFanOut(workflow.Start, renewableNode, electricVehicleNode, carbonNode)
	eb.AddFanIn(gatherNode, renewableNode, electricVehicleNode, carbonNode)
	eb.Add(gatherNode, formatNode)
	eb.Add(formatNode, synthNode)

	return workflowagent.New(workflowagent.Config{
		Name:        "research_and_synthesis_pipeline",
		Description: "runs three researchers in parallel and synthesizes their findings",
		Edges:       eb.Build(),
		// Register the wrapped LLM agents so the runner can resolve each
		// event's author; otherwise it logs a harmless "Event from an
		// unknown agent" on every turn.
		SubAgents: []agent.Agent{renewableAgent, electricVehicleAgent, carbonAgent, synthAgent},
	})
}

func main() {
	ctx := context.Background()

	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		log.Fatalf("GOOGLE_API_KEY is required to run this sample")
	}

	m, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{APIKey: apiKey})
	if err != nil {
		log.Fatalf("failed to create model: %v", err)
	}

	rootAgent, err := newResearchPipeline(m)
	if err != nil {
		log.Fatalf("failed to build research pipeline: %v", err)
	}

	log.Printf("research pipeline ready — send any message to kick off the run")

	cfg := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	}
	l := full.NewLauncher()
	if err := l.Execute(ctx, cfg, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
