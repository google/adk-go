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

package utils_test

import (
	"context"
	"errors"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
)

func TestErrRelay(t *testing.T) {
	boom := errors.New("boom")
	newAgent := func(name string) agent.Agent {
		a, err := agent.New(agent.Config{Name: name})
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	inner := newAgent("inner")
	// The wrapper sequentialagent.New returns must count as the agent it
	// wraps, since ctx.Agent() holds the wrapped one while it runs.
	seq, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq"}})
	if err != nil {
		t.Fatal(err)
	}
	runCtx := func(ctx context.Context, a agent.Agent, invocationID string) agent.InvocationContext {
		return icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{Agent: a, InvocationID: invocationID})
	}

	var got []error
	base := utils.WithErrRelay(t.Context(), "inv", seq, func(err error) { got = append(got, err) })

	// Another agent with the bound agent's name does not report, and neither
	// does the bound agent in another invocation.
	utils.ReportErr(runCtx(base, newAgent("seq"), "inv"), boom)
	utils.ReportErr(runCtx(base, seq, "other"), boom)
	if len(got) != 0 {
		t.Fatalf("ReportErr() through a relay bound elsewhere reported %v, want nothing", got)
	}

	// Only the bound agent passes the relay on.
	innerCtx := utils.PassErrRelay(runCtx(base, newAgent("seq"), "inv"), inner)
	utils.ReportErr(runCtx(innerCtx, inner, "inv"), boom)
	if len(got) != 0 {
		t.Fatalf("ReportErr() after a pass by an agent the relay is not bound to reported %v, want nothing", got)
	}
	innerCtx = utils.PassErrRelay(runCtx(base, seq, "inv"), inner)
	utils.ReportErr(runCtx(innerCtx, seq, "inv"), boom)
	if len(got) != 0 {
		t.Fatalf("ReportErr() by the agent that passed the relay on reported %v, want nothing", got)
	}
	utils.ReportErr(runCtx(innerCtx, inner, "inv"), boom)
	if len(got) != 1 || got[0] != boom {
		t.Errorf("ReportErr() by the agent the relay was passed to reported %v, want [%v]", got, boom)
	}

	// An agent that does not expose its State matches no relay, not even one
	// bound to another such agent.
	got = nil
	bare := embeddedAgent{inner}
	bareCtx := utils.WithErrRelay(t.Context(), "inv", bare, func(err error) { got = append(got, err) })
	utils.ReportErr(runCtx(bareCtx, embeddedAgent{newAgent("other")}, "inv"), boom)
	if len(got) != 0 {
		t.Errorf("ReportErr() between agents without a State reported %v, want nothing", got)
	}
}

// embeddedAgent embeds an agent without its State.
type embeddedAgent struct{ agent.Agent }
