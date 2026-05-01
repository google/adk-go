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

package builtin

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// DebugLoggingConfig configures NewDebugLogging.
type DebugLoggingConfig struct {
	// Name overrides the default plugin name.
	Name string

	// Out overrides the destination writer. Default: os.Stdout.
	Out io.Writer

	// Plain disables ASCII markers ("USER MESSAGE", "AGENT RUN", etc).
	// When true, output is text-only with no decorative prefix.
	Plain bool
}

// NewDebugLogging builds a plugin that prints a human-readable trace
// of every callback to a writer (stdout by default). Useful for local
// development and live debugging — not intended as a structured-logs
// replacement (see LoggingPlugin for that).
//
// Mirrors adk-python's DebugLoggingPlugin, with the same labelled
// sections at each lifecycle hook.
func NewDebugLogging(cfg DebugLoggingConfig) (*plugin.Plugin, error) {
	name := cfg.Name
	if name == "" {
		name = "debug_logging"
	}
	out := cfg.Out
	if out == nil {
		out = os.Stdout
	}

	prefix := func(label string) string {
		if cfg.Plain {
			return label + ": "
		}
		return "[ADK " + label + "] "
	}

	var mu sync.Mutex
	emit := func(label, body string) {
		mu.Lock()
		defer mu.Unlock()
		fmt.Fprintln(out, prefix(label)+body)
	}

	contentText := func(c *genai.Content) string {
		if c == nil {
			return ""
		}
		var b strings.Builder
		for _, p := range c.Parts {
			if p == nil {
				continue
			}
			if p.Text != "" {
				if b.Len() > 0 {
					b.WriteString(" | ")
				}
				b.WriteString(p.Text)
			}
			if p.FunctionCall != nil {
				if b.Len() > 0 {
					b.WriteString(" | ")
				}
				fmt.Fprintf(&b, "fn_call(%s id=%s args=%v)",
					p.FunctionCall.Name, p.FunctionCall.ID, p.FunctionCall.Args)
			}
			if p.FunctionResponse != nil {
				if b.Len() > 0 {
					b.WriteString(" | ")
				}
				fmt.Fprintf(&b, "fn_response(%s id=%s)",
					p.FunctionResponse.Name, p.FunctionResponse.ID)
			}
		}
		return b.String()
	}

	return plugin.New(plugin.Config{
		Name: name,

		OnUserMessageCallback: func(ic agent.InvocationContext, msg *genai.Content) (*genai.Content, error) {
			emit("USER", fmt.Sprintf("inv=%s session=%s text=%q", ic.InvocationID(), ic.Session().ID(), contentText(msg)))
			return nil, nil
		},
		BeforeRunCallback: func(ic agent.InvocationContext) (*genai.Content, error) {
			emit("RUN_START", fmt.Sprintf("inv=%s agent=%s", ic.InvocationID(), ic.Agent().Name()))
			return nil, nil
		},
		AfterRunCallback: func(ic agent.InvocationContext) {
			emit("RUN_END", fmt.Sprintf("inv=%s agent=%s", ic.InvocationID(), ic.Agent().Name()))
		},
		OnEventCallback: func(ic agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			if ev == nil {
				return nil, nil
			}
			emit("EVENT", fmt.Sprintf("id=%s author=%s partial=%t text=%q",
				ev.ID, ev.Author, ev.Partial, contentText(ev.Content)))
			return nil, nil
		},
		BeforeAgentCallback: func(cctx agent.CallbackContext) (*genai.Content, error) {
			emit("AGENT_BEFORE", fmt.Sprintf("agent=%s inv=%s", cctx.AgentName(), cctx.InvocationID()))
			return nil, nil
		},
		AfterAgentCallback: func(cctx agent.CallbackContext) (*genai.Content, error) {
			emit("AGENT_AFTER", fmt.Sprintf("agent=%s inv=%s", cctx.AgentName(), cctx.InvocationID()))
			return nil, nil
		},
		BeforeModelCallback: func(cctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
			ntools := 0
			if req != nil {
				ntools = len(req.Tools)
			}
			emit("MODEL_BEFORE", fmt.Sprintf("agent=%s model=%s tools=%d",
				cctx.AgentName(), modelName(req), ntools))
			return nil, nil
		},
		AfterModelCallback: func(cctx agent.CallbackContext, resp *model.LLMResponse, _ error) (*model.LLMResponse, error) {
			body := ""
			if resp != nil {
				body = contentText(resp.Content)
			}
			emit("MODEL_AFTER", fmt.Sprintf("agent=%s text=%q", cctx.AgentName(), body))
			return nil, nil
		},
		OnModelErrorCallback: func(cctx agent.CallbackContext, _ *model.LLMRequest, err error) (*model.LLMResponse, error) {
			emit("MODEL_ERROR", fmt.Sprintf("agent=%s err=%s", cctx.AgentName(), err.Error()))
			return nil, nil
		},
		BeforeToolCallback: func(tctx tool.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
			emit("TOOL_BEFORE", fmt.Sprintf("agent=%s tool=%s call_id=%s args=%v",
				tctx.AgentName(), t.Name(), tctx.FunctionCallID(), args))
			return nil, nil
		},
		AfterToolCallback: func(tctx tool.Context, t tool.Tool, _ map[string]any, result map[string]any, runErr error) (map[string]any, error) {
			if runErr != nil {
				emit("TOOL_AFTER_ERR", fmt.Sprintf("agent=%s tool=%s call_id=%s err=%s",
					tctx.AgentName(), t.Name(), tctx.FunctionCallID(), runErr.Error()))
			} else {
				emit("TOOL_AFTER", fmt.Sprintf("agent=%s tool=%s call_id=%s result=%v",
					tctx.AgentName(), t.Name(), tctx.FunctionCallID(), result))
			}
			return nil, nil
		},
		OnToolErrorCallback: func(tctx tool.Context, t tool.Tool, _ map[string]any, err error) (map[string]any, error) {
			emit("TOOL_ERROR", fmt.Sprintf("agent=%s tool=%s err=%s",
				tctx.AgentName(), t.Name(), err.Error()))
			return nil, nil
		},
	})
}

func modelName(req *model.LLMRequest) string {
	if req == nil {
		return ""
	}
	return req.Model
}
