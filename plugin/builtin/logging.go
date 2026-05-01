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
	"log/slog"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// LoggingConfig configures NewLogging.
type LoggingConfig struct {
	// Name overrides the default plugin name "logging".
	Name string

	// Logger overrides the default *slog.Logger (uses slog.Default
	// when nil).
	Logger *slog.Logger

	// IncludeContent, when true, logs full message content. Default
	// false to keep logs lean and avoid leaking sensitive prompts.
	IncludeContent bool
}

// NewLogging builds a structured-logging plugin that emits a slog
// record at every callback hook (user message, before/after run,
// before/after model, before/after tool, on event). Mirrors a
// streamlined adk-python LoggingPlugin.
//
// Each record carries app_name, session_id, invocation_id, and the
// hook-specific dimensions (model name, tool name, event author, etc.)
// as structured attributes so a downstream slog handler can route
// records to JSON, otel, or any custom sink.
func NewLogging(cfg LoggingConfig) (*plugin.Plugin, error) {
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}
	name := cfg.Name
	if name == "" {
		name = "logging"
	}

	commonAttrs := func(cctx agent.CallbackContext) []any {
		return []any{
			slog.String("plugin", name),
			slog.String("app", cctx.AppName()),
			slog.String("session", cctx.SessionID()),
			slog.String("invocation", cctx.InvocationID()),
			slog.String("agent", cctx.AgentName()),
		}
	}
	icAttrs := func(ic agent.InvocationContext) []any {
		return []any{
			slog.String("plugin", name),
			slog.String("app", ic.Session().AppName()),
			slog.String("session", ic.Session().ID()),
			slog.String("invocation", ic.InvocationID()),
			slog.String("agent", ic.Agent().Name()),
		}
	}

	logIfContent := func(parts []*genai.Part) string {
		if !cfg.IncludeContent {
			return ""
		}
		out := ""
		for _, p := range parts {
			if p == nil {
				continue
			}
			if p.Text != "" {
				if out != "" {
					out += "\n"
				}
				out += p.Text
			}
		}
		return out
	}

	cfg2 := plugin.Config{
		Name: name,

		OnUserMessageCallback: func(ic agent.InvocationContext, msg *genai.Content) (*genai.Content, error) {
			attrs := icAttrs(ic)
			if msg != nil {
				attrs = append(attrs, slog.String("role", msg.Role))
				if t := logIfContent(msg.Parts); t != "" {
					attrs = append(attrs, slog.String("text", t))
				}
			}
			logger.Info("user_message", attrs...)
			return nil, nil
		},

		BeforeRunCallback: func(ic agent.InvocationContext) (*genai.Content, error) {
			logger.Info("before_run", icAttrs(ic)...)
			return nil, nil
		},

		AfterRunCallback: func(ic agent.InvocationContext) {
			logger.Info("after_run", icAttrs(ic)...)
		},

		OnEventCallback: func(ic agent.InvocationContext, ev *session.Event) (*session.Event, error) {
			attrs := icAttrs(ic)
			if ev != nil {
				attrs = append(attrs,
					slog.String("event_id", ev.ID),
					slog.String("event_author", ev.Author),
					slog.Bool("partial", ev.Partial),
				)
			}
			logger.Debug("event", attrs...)
			return nil, nil
		},

		BeforeAgentCallback: func(cctx agent.CallbackContext) (*genai.Content, error) {
			logger.Info("before_agent", commonAttrs(cctx)...)
			return nil, nil
		},
		AfterAgentCallback: func(cctx agent.CallbackContext) (*genai.Content, error) {
			logger.Info("after_agent", commonAttrs(cctx)...)
			return nil, nil
		},

		BeforeModelCallback: func(cctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
			attrs := commonAttrs(cctx)
			if req != nil {
				attrs = append(attrs, slog.String("model", req.Model))
			}
			logger.Info("before_model", attrs...)
			return nil, nil
		},
		AfterModelCallback: func(cctx agent.CallbackContext, resp *model.LLMResponse, _ error) (*model.LLMResponse, error) {
			attrs := commonAttrs(cctx)
			if resp != nil {
				attrs = append(attrs,
					slog.Bool("partial", resp.Partial),
					slog.String("finish_reason", string(resp.FinishReason)),
				)
				if resp.UsageMetadata != nil {
					attrs = append(attrs,
						slog.Int("prompt_tokens", int(resp.UsageMetadata.PromptTokenCount)),
						slog.Int("completion_tokens", int(resp.UsageMetadata.CandidatesTokenCount)),
					)
				}
			}
			logger.Info("after_model", attrs...)
			return nil, nil
		},
		OnModelErrorCallback: func(cctx agent.CallbackContext, _ *model.LLMRequest, err error) (*model.LLMResponse, error) {
			attrs := append(commonAttrs(cctx), slog.String("err", err.Error()))
			logger.Error("on_model_error", attrs...)
			return nil, nil
		},

		BeforeToolCallback: func(tctx tool.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
			attrs := []any{
				slog.String("plugin", name),
				slog.String("agent", tctx.AgentName()),
				slog.String("invocation", tctx.InvocationID()),
				slog.String("tool", t.Name()),
				slog.String("call_id", tctx.FunctionCallID()),
			}
			logger.Info("before_tool", attrs...)
			return nil, nil
		},
		AfterToolCallback: func(tctx tool.Context, t tool.Tool, _, _ map[string]any, runErr error) (map[string]any, error) {
			attrs := []any{
				slog.String("plugin", name),
				slog.String("agent", tctx.AgentName()),
				slog.String("invocation", tctx.InvocationID()),
				slog.String("tool", t.Name()),
				slog.String("call_id", tctx.FunctionCallID()),
			}
			if runErr != nil {
				attrs = append(attrs, slog.String("err", runErr.Error()))
			}
			logger.Info("after_tool", attrs...)
			return nil, nil
		},
		OnToolErrorCallback: func(tctx tool.Context, t tool.Tool, _ map[string]any, err error) (map[string]any, error) {
			logger.Error("on_tool_error",
				slog.String("plugin", name),
				slog.String("tool", t.Name()),
				slog.String("err", err.Error()),
			)
			return nil, nil
		},
	}

	if os.Getenv("ADK_LOGGING_DISABLED") != "" {
		return plugin.New(plugin.Config{Name: name})
	}
	return plugin.New(cfg2)
}
