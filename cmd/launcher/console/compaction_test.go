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

package console

import (
	"context"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// countingSummarizer records that compaction reached the summarizer, and can be
// made to fail.
type countingSummarizer struct {
	mu    sync.Mutex
	calls int
	err   error
}

func (c *countingSummarizer) SummarizeEvents(context.Context, []*session.Event) (compaction.SummarizeResult, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls++
	if c.err != nil {
		return compaction.SummarizeResult{}, c.err
	}
	return compaction.SummarizeResult{Content: genai.NewContentFromText("a summary", genai.RoleModel)}, nil
}

func (c *countingSummarizer) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls
}

// runConsole drives the console launcher over a scripted stdin and returns what
// it printed.
//
// The launcher reads os.Stdin directly, so the pipe is swapped in for the
// duration. Closing the write end ends the loop the way EOF does in a terminal.
func runConsole(t *testing.T, config *launcher.Config, input string) string {
	t.Helper()

	inR, inW, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe() error = %v", err)
	}
	outR, outW, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe() error = %v", err)
	}
	oldIn, oldOut := os.Stdin, os.Stdout
	os.Stdin, os.Stdout = inR, outW
	t.Cleanup(func() { os.Stdin, os.Stdout = oldIn, oldOut })

	printed := make(chan string, 1)
	go func() {
		var sb strings.Builder
		buf := make([]byte, 4096)
		for {
			n, err := outR.Read(buf)
			sb.Write(buf[:n])
			if err != nil {
				break
			}
		}
		printed <- sb.String()
	}()

	if _, err := inW.WriteString(input); err != nil {
		t.Fatalf("writing stdin: %v", err)
	}
	if err := inW.Close(); err != nil {
		t.Fatalf("closing stdin: %v", err)
	}

	l := NewLauncher()
	if _, err := l.Parse(nil); err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	runErr := l.Run(t.Context(), config)
	if err := outW.Close(); err != nil {
		t.Fatalf("closing stdout: %v", err)
	}
	out := <-printed
	if runErr != nil {
		t.Fatalf("Run() error = %v. Output:\n%s", runErr, out)
	}
	return out
}

func testConfig(t *testing.T, cfg *compaction.Config) *launcher.Config {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{
		Name: "echo",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(cc agent.Context) (*genai.Content, error) { return cc.UserContent(), nil },
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	return &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(a),
		SessionService: session.InMemoryService(),
		Compaction:     cfg,
	}
}

// TestConsoleActuallyRunsCompaction pins that the compaction config this
// surface accepts reaches the runner it builds.
//
// The console builds a runner.Config inside Run, and deleting the line that
// copies Compaction into it left the whole suite green: the other
// tests in this package build a runner directly and never exercise the
// launcher's own wiring. Counting summarizer calls is what distinguishes
// "compaction ran" from "this surface accepts a compaction config and does
// nothing with it".
func TestConsoleActuallyRunsCompaction(t *testing.T) {
	summarizer := &countingSummarizer{}
	config := testConfig(t, &compaction.Config{CompactionInterval: 1, Summarizer: summarizer})

	runConsole(t, config, "hello\n")

	if got := summarizer.count(); got == 0 {
		t.Error("the summarizer was never called, so the console accepts a compaction config and does nothing with it")
	}
}

// TestConsoleSurvivesACompactionFailure pins that a compaction failure does not
// surface as a failed turn.
//
// Compaction runs after the agent has answered and after its events are
// stored, so a failure there means only that a later prompt will be larger.
// Printing AGENT_ERROR underneath an answer the user has already read says the
// turn failed when it did not.
func TestConsoleSurvivesACompactionFailure(t *testing.T) {
	summarizer := &countingSummarizer{err: errSummarizerUnavailable}
	config := testConfig(t, &compaction.Config{CompactionInterval: 1, Summarizer: summarizer})

	out := runConsole(t, config, "hello\n")

	if summarizer.count() == 0 {
		t.Fatal("the summarizer was never called, so this test is not exercising the failure path")
	}
	if strings.Contains(out, "AGENT_ERROR") {
		t.Errorf("a compaction failure was reported as a failed turn:\n%s", out)
	}
}

var errSummarizerUnavailable = errors.New("summarizer unavailable")
