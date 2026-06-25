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

package loggingplugin

import (
	"context"
	"fmt"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// New creates an instance of the logging plugin.
//
// This plugin helps print all critical events in the console. It is not a
// replacement of existing logging in ADK. It rather helps terminal based
// debugging by showing all logs in the console, and serves as a simple demo for
// everyone to leverage when developing new plugins.
//
// This plugin helps users track the invocation status by logging:
// - User messages and invocation context
// - Agent execution flow
// - LLM requests and responses
// - Tool calls with arguments and results
// - Events and final responses
// - Errors during model and tool execution
func New(name string) (*plugin.Plugin, error) {
	if name == "" {
		name = "logging_plugin"
	}
	p := &loggingPlugin{name: name}
	return plugin.New(plugin.Config{
		Name:                  name,
		OnUserMessageCallback: p.onUserMessage,
		BeforeRunCallback:     p.beforeRun,
		OnEventCallback:       p.onEvent,
		AfterRunCallback:      p.afterRun,
		BeforeAgentCallback:   p.beforeAgent,
		AfterAgentCallback:    p.afterAgent,
		BeforeModelCallback:   p.beforeModel,
		AfterModelCallback:    p.afterModel,
		OnModelErrorCallback:  p.onModelError,
		BeforeToolCallback:    p.beforeTool,
		AfterToolCallback:     p.afterTool,
		OnToolErrorCallback:   p.onToolError,
	})
}

// MustNew is like New but panics if there is an error.
func MustNew(name string) *plugin.Plugin {
	p, err := New(name)
	if err != nil {
		panic(err)
	}
	return p
}

type loggingPlugin struct {
	name string
}

func (p *loggingPlugin) log(msg string) {
	// ANSI color codes: \033[90m for grey, \033[0m to reset
	fmt.Printf("\033[90m[%s] %s\033[0m\n", p.name, msg)
}

func (p *loggingPlugin) formatContent(content *genai.Content, maxLength int) string {
	if content == nil || len(content.Parts) == 0 {
		return "None"
	}

	var parts []string
	for _, part := range content.Parts {
		if part.Text != "" {
			text := strings.TrimSpace(part.Text)
			if len(text) > maxLength {
				text = text[:maxLength] + "..."
			}
			parts = append(parts, fmt.Sprintf("text: '%s'", text))
		} else if part.FunctionCall != nil {
			parts = append(parts, fmt.Sprintf("function_call: %s", part.FunctionCall.Name))
		} else if part.FunctionResponse != nil {
			parts = append(parts, fmt.Sprintf("function_response: %s", part.FunctionResponse.Name))
		} else if part.CodeExecutionResult != nil {
			parts = append(parts, "code_execution_result")
		} else {
			parts = append(parts, "other_part")
		}
	}
	return strings.Join(parts, " | ")
}

func (p *loggingPlugin) formatArgs(args map[string]any, maxLength int) string {
	if len(args) == 0 {
		return "{}"
	}
	formatted := fmt.Sprintf("%v", args)
	if len(formatted) > maxLength {
		formatted = formatted[:maxLength] + "...}"
	}
	return formatted
}

func (p *loggingPlugin) onUserMessage(ctx context.Context, invCleanCtx agent.InvocationContext, userMessage *genai.Content) (*genai.Content, error) {
	p.log("🚀 USER MESSAGE RECEIVED")
	p.log(fmt.Sprintf("   Invocation ID: %s", invCleanCtx.InvocationID()))
	p.log(fmt.Sprintf("   Session ID: %s", invCleanCtx.Session().ID()))
	p.log(fmt.Sprintf("   User ID: %s", invCleanCtx.Session().UserID()))
	p.log(fmt.Sprintf("   App Name: %s", invCleanCtx.Session().AppName()))
	agentName := "Unknown"
	if invCleanCtx.Agent() != nil {
		agentName = invCleanCtx.Agent().Name()
	}
	p.log(fmt.Sprintf("   Root Agent: %s", agentName))
	p.log(fmt.Sprintf("   User Content: %s", p.formatContent(userMessage, 200)))
	if invCleanCtx.Branch() != "" {
		p.log(fmt.Sprintf("   Branch: %s", invCleanCtx.Branch()))
	}
	return nil, nil
}

func (p *loggingPlugin) beforeRun(ctx context.Context, invCleanCtx agent.InvocationContext) (*genai.Content, error) {
	p.log("🏃 INVOCATION STARTING")
	p.log(fmt.Sprintf("   Invocation ID: %s", invCleanCtx.InvocationID()))
	agentName := "Unknown"
	if invCleanCtx.Agent() != nil {
		agentName = invCleanCtx.Agent().Name()
	}
	p.log(fmt.Sprintf("   Starting Agent: %s", agentName))
	return nil, nil
}

func (p *loggingPlugin) onEvent(ctx context.Context, invCleanCtx agent.InvocationContext, event *session.Event) (*session.Event, error) {
	p.log("📢 EVENT YIELDED")
	p.log(fmt.Sprintf("   Event ID: %s", event.ID))
	p.log(fmt.Sprintf("   Author: %s", event.Author))
	p.log(fmt.Sprintf("   Content: %s", p.formatContent(event.Content, 200)))
	p.log(fmt.Sprintf("   Final Response: %v", event.IsFinalResponse()))

	var funcCalls []string
	var funcResponses []string

	if event.Content != nil {
		for _, part := range event.Content.Parts {
			if part.FunctionCall != nil {
				funcCalls = append(funcCalls, part.FunctionCall.Name)
			}
			if part.FunctionResponse != nil {
				funcResponses = append(funcResponses, part.FunctionResponse.Name)
			}
		}
	}

	if len(funcCalls) > 0 {
		p.log(fmt.Sprintf("   Function Calls: %v", funcCalls))
	}
	if len(funcResponses) > 0 {
		p.log(fmt.Sprintf("   Function Responses: %v", funcResponses))
	}
	if len(event.LongRunningToolIDs) > 0 {
		p.log(fmt.Sprintf("   Long Running Tools: %v", event.LongRunningToolIDs))
	}

	return nil, nil
}

func (p *loggingPlugin) afterRun(ctx context.Context, invCleanCtx agent.InvocationContext) {
	p.log("✅ INVOCATION COMPLETED")
	p.log(fmt.Sprintf("   Invocation ID: %s", invCleanCtx.InvocationID()))
	agentName := "Unknown"
	if invCleanCtx.Agent() != nil {
		agentName = invCleanCtx.Agent().Name()
	}
	p.log(fmt.Sprintf("   Final Agent: %s", agentName))
}

func (p *loggingPlugin) beforeAgent(ctx context.Context, invCleanCtx agent.Context) (*genai.Content, error) {
	p.log("🤖 AGENT STARTING")
	p.log(fmt.Sprintf("   Agent Name: %s", invCleanCtx.AgentName()))
	p.log(fmt.Sprintf("   Invocation ID: %s", invCleanCtx.InvocationID()))
	if invCleanCtx.Branch() != "" {
		p.log(fmt.Sprintf("   Branch: %s", invCleanCtx.Branch()))
	}
	return nil, nil
}

func (p *loggingPlugin) afterAgent(ctx context.Context, invCleanCtx agent.Context) (*genai.Content, error) {
	p.log("🤖 AGENT COMPLETED")
	p.log(fmt.Sprintf("   Agent Name: %s", invCleanCtx.AgentName()))
	p.log(fmt.Sprintf("   Invocation ID: %s", invCleanCtx.InvocationID()))
	return nil, nil
}

func (p *loggingPlugin) beforeModel(ctx context.Context, invCleanCtx agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	p.log("🧠 LLM REQUEST")
	modelName := "default"
	if req.Model != "" {
		modelName = req.Model
	}
	p.log(fmt.Sprintf("   Model: %s", modelName))
	p.log(fmt.Sprintf("   Agent: %s", invCleanCtx.AgentName()))

	if req.Config != nil && req.Config.SystemInstruction != nil && len(req.Config.SystemInstruction.Parts) > 0 {
		// Assuming SystemInstruction is a Content object with parts
		sysInstruction := ""
		for _, part := range req.Config.SystemInstruction.Parts {
			sysInstruction += part.Text
		}
		if len(sysInstruction) > 200 {
			sysInstruction = sysInstruction[:200] + "..."
		}
		p.log(fmt.Sprintf("   System Instruction: '%s'", sysInstruction))
	}

	if len(req.Tools) > 0 {
		var toolNames []string
		for name := range req.Tools {
			toolNames = append(toolNames, name)
		}
		p.log(fmt.Sprintf("   Available Tools: %v", toolNames))
	}

	return nil, nil
}

func (p *loggingPlugin) afterModel(ctx context.Context, invCleanCtx agent.Context, resp *model.LLMResponse, err error) (*model.LLMResponse, error) {
	p.log("🧠 LLM RESPONSE")
	p.log(fmt.Sprintf("   Agent: %s", invCleanCtx.AgentName()))

	// If error passed in, log it
	if err != nil {
		p.log(fmt.Sprintf("   ❌ ERROR - %v", err))
		return nil, nil // Return nil, nil to propagate original error
	}

	if resp != nil {
		if resp.ErrorCode != "" {
			p.log(fmt.Sprintf("   ❌ ERROR - Code: %s", resp.ErrorCode))
			p.log(fmt.Sprintf("   Error Message: %s", resp.ErrorMessage))
		} else {
			p.log(fmt.Sprintf("   Content: %s", p.formatContent(resp.Content, 200)))
			if resp.Partial {
				p.log(fmt.Sprintf("   Partial: %v", resp.Partial))
			}
			// TurnComplete is a boolean in Go model
			p.log(fmt.Sprintf("   Turn Complete: %v", resp.TurnComplete))
		}

		if resp.UsageMetadata != nil {
			p.log(fmt.Sprintf("   Token Usage - Input: %d, Output: %d",
				resp.UsageMetadata.PromptTokenCount, resp.UsageMetadata.CandidatesTokenCount))
		}
	}

	return nil, nil
}

func (p *loggingPlugin) onModelError(ctx context.Context, invCleanCtx agent.Context, req *model.LLMRequest, err error) (*model.LLMResponse, error) {
	p.log("🧠 LLM ERROR")
	p.log(fmt.Sprintf("   Agent: %s", invCleanCtx.AgentName()))
	p.log(fmt.Sprintf("   Error: %v", err))
	return nil, nil
}

func (p *loggingPlugin) beforeTool(ctx context.Context, invCleanCtx agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
	p.log("🔧 TOOL STARTING")
	p.log(fmt.Sprintf("   Tool Name: %s", t.Name()))
	p.log(fmt.Sprintf("   Agent: %s", invCleanCtx.AgentName()))
	p.log(fmt.Sprintf("   Function Call ID: %s", invCleanCtx.FunctionCallID()))
	p.log(fmt.Sprintf("   Arguments: %s", p.formatArgs(args, 300)))
	return nil, nil
}

func (p *loggingPlugin) afterTool(ctx context.Context, invCleanCtx agent.Context, t tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
	p.log("🔧 TOOL COMPLETED")
	p.log(fmt.Sprintf("   Tool Name: %s", t.Name()))
	p.log(fmt.Sprintf("   Agent: %s", invCleanCtx.AgentName()))
	p.log(fmt.Sprintf("   Function Call ID: %s", invCleanCtx.FunctionCallID()))
	if err != nil {
		p.log(fmt.Sprintf("   Error: %v", err))
	} else {
		p.log(fmt.Sprintf("   Result: %s", p.formatArgs(result, 300)))
	}
	return nil, nil
}

func (p *loggingPlugin) onToolError(ctx context.Context, invCleanCtx agent.Context, t tool.Tool, args map[string]any, err error) (map[string]any, error) {
	p.log("🔧 TOOL ERROR")
	p.log(fmt.Sprintf("   Tool Name: %s", t.Name()))
	p.log(fmt.Sprintf("   Agent: %s", invCleanCtx.AgentName()))
	p.log(fmt.Sprintf("   Function Call ID: %s", invCleanCtx.FunctionCallID()))
	p.log(fmt.Sprintf("   Arguments: %s", p.formatArgs(args, 300)))
	p.log(fmt.Sprintf("   Error: %v", err))
	return nil, nil
}
