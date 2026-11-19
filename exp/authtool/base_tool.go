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

package authtool

import (
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolauth"
	"google.golang.org/adk/v2/tool/toolutils"
)

// BaseHandler is the inner handler for a custom authenticated tool.
//
// Call runs the tool. args holds the arguments of the call as the model sent
// them, unchecked against the input schema, and cred and authErr are what
// [Func] receives. Call's result is the tool's result, and its error is
// returned unchanged. Call may run for several calls at once.
//
// This is the _run_async_impl a subclass of adk-python's BaseAuthenticatedTool
// implements (tools/base_authenticated_tool.py), which adk-python marks
// experimental.
//
// EXPERIMENTAL: BaseHandler is experimental; its API and behavior may change or be removed in a later version.
type BaseHandler interface {
	Call(ctx agent.Context, args any, cred *toolauth.Credential, authErr error) (map[string]any, error)
}

// NewBaseTool creates a custom authenticated tool wrapping handler.
//
// cfg's embedded [functiontool.Config] gives the tool's name, description,
// input schema and output schema, which the declaration carries as given, and
// its IsLongRunning, ResponseScheduling and Sequential, which
// mean what they mean for [functiontool.New]. The tool gets the credential,
// asks the user for it and returns errors as the tool [New] builds does, then
// calls handler.
//
// NewBaseTool returns (nil, err) when a schema of cfg does not resolve, when
// cfg sets RequireConfirmation or RequireConfirmationProvider (wrap the tool's
// toolset with [tool.WithConfirmation] instead), and in the cases [New]
// describes for ResponseForAuthRequired and AuthConfig.
//
// This is adk-python's BaseAuthenticatedTool
// (tools/base_authenticated_tool.py), which adk-python marks experimental.
//
// EXPERIMENTAL: NewBaseTool is experimental; its API and behavior may change or be removed in a later version.
func NewBaseTool(cfg Config, handler BaseHandler) (tool.Tool, error) {
	if cfg.RequireConfirmation || cfg.RequireConfirmationProvider != nil {
		return nil, fmt.Errorf("authtool: tool %q: NewBaseTool does not support RequireConfirmation or RequireConfirmationProvider: %w", cfg.Name, functiontool.ErrInvalidArgument)
	}
	for _, s := range []*jsonschema.Schema{cfg.InputSchema, cfg.OutputSchema} {
		if s == nil {
			continue
		}
		if _, err := s.Resolve(nil); err != nil {
			return nil, fmt.Errorf("authtool: tool %q: %w", cfg.Name, err)
		}
	}
	return newAuthTool(&baseTool{cfg: cfg.Config, handler: handler}, cfg)
}

// baseTool calls a BaseHandler with the credential authTool put on the
// call's context.
type baseTool struct {
	cfg     functiontool.Config
	handler BaseHandler
}

func (t *baseTool) Name() string { return t.cfg.Name }

func (t *baseTool) Description() string { return t.cfg.Description }

func (t *baseTool) IsLongRunning() bool { return t.cfg.IsLongRunning }

func (t *baseTool) ResponseScheduling() genai.FunctionResponseScheduling {
	return t.cfg.ResponseScheduling
}

func (t *baseTool) RunsSequentially() bool { return t.cfg.Sequential }

func (t *baseTool) Declaration() *genai.FunctionDeclaration {
	decl := &genai.FunctionDeclaration{Name: t.cfg.Name, Description: t.cfg.Description}
	// Assigned only when set, so that a missing schema stays a nil interface.
	if t.cfg.InputSchema != nil {
		decl.ParametersJsonSchema = t.cfg.InputSchema
	}
	if t.cfg.OutputSchema != nil {
		decl.ResponseJsonSchema = t.cfg.OutputSchema
	}
	return decl
}

func (t *baseTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

func (t *baseTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	cred, authErr := callAuthFrom(ctx)
	return t.handler.Call(ctx, args, cred, authErr)
}

// RunRaw is Run, with a nil result as an untyped nil, as a function tool's
// RunRaw returns it.
func (t *baseTool) RunRaw(ctx agent.Context, args any) (any, error) {
	result, err := t.Run(ctx, args)
	if result == nil {
		return nil, err
	}
	return result, err
}
