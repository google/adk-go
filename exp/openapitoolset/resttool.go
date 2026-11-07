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

package openapitoolset

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolauth"
	"google.golang.org/adk/v2/tool/toolutils"
)

// maxErrorBodyBytes caps how much of a non-2xx response body is read into the
// error returned to the model.
const maxErrorBodyBytes = 64 << 10

// Endpoint is where a REST API tool sends its request: Method to the URL
// BaseURL followed by Path. Path is the OpenAPI path template, whose {name}
// segments are filled in from the path parameters.
type Endpoint struct {
	BaseURL string
	Path    string
	Method  string
}

// RESTAPIToolConfig configures a [RESTAPITool] made from one OpenAPI
// operation on its own, without a spec around it.
type RESTAPIToolConfig struct {
	// Name is the tool's name. Only its first 60 characters are used, since
	// Gemini limits function names to 64. Required.
	Name        string
	Description string
	Endpoint    Endpoint
	// Operation is an OpenAPI Operation object as decoded from JSON. It is
	// used as given: references in it are not resolved, its operationId does
	// not name the tool, and its security requirement is not consulted. Nil
	// is an operation with no parameters and no request body.
	Operation map[string]any

	// AuthScheme and AuthCredential authenticate the request. See
	// [Config.AuthScheme] and [Config.AuthCredential]. An operation built here
	// has no security of its own, so these are the only source of auth for it.
	AuthScheme     *toolauth.Scheme
	AuthCredential *toolauth.Credential

	// HTTPClient sends the tool's request. When nil, a client with bounded
	// timeouts is used. See [Config.HTTPClient].
	HTTPClient *http.Client
	// HeaderProvider adds headers to each request. See [Config.HeaderProvider].
	HeaderProvider func(agent.ReadonlyContext) (map[string]string, error)
	// BlockPrivateNetworks refuses requests to private, loopback, link-local
	// and similar targets. See [Config.BlockPrivateNetworks].
	BlockPrivateNetworks bool
}

// RESTAPITool is a tool for one operation of a REST API. Its declaration
// offers the operation's parameters and request body to the model as
// arguments, named as described in [New], and calling it sends the HTTP
// request and returns the decoded response.
//
// A RESTAPITool does not change after it is made and is safe for concurrent
// use.
type RESTAPITool struct {
	name        string
	description string
	endpoint    Endpoint
	params      []*apiParameter
	// body describes how the request body is sent, or nil when there is none.
	body *requestBodyInfo
	// authScheme is the scheme the operation's security requirement names in
	// the spec, or the toolset-level override, or nil.
	authScheme *toolauth.Scheme
	// authCredential is the credential to authenticate with, or nil.
	authCredential *toolauth.Credential
	// parameters is the JSON schema of the arguments.
	parameters map[string]any

	client         *http.Client
	headerProvider func(agent.ReadonlyContext) (map[string]string, error)
	blockPrivate   bool
}

// NewRESTAPITool returns a tool for the operation in cfg. It fails when the
// operation cannot be parsed, for example when a schema in it is the boolean
// false.
func NewRESTAPITool(cfg RESTAPIToolConfig) (*RESTAPITool, error) {
	if cfg.Name == "" {
		return nil, errors.New("openapitoolset: RESTAPIToolConfig.Name is required")
	}
	op, err := objectFromMap(cfg.Operation)
	if err != nil {
		return nil, fmt.Errorf("openapitoolset: operation: %w", err)
	}
	params, body, err := parseOperation(op, false)
	if err != nil {
		return nil, fmt.Errorf("openapitoolset: operation: %w", err)
	}
	operationID, _ := op.get("operationId").(string)
	t := newRESTAPITool(truncate(cfg.Name, maxToolNameLength), cfg.Description, cfg.Endpoint, operationID, params, body, cfg.AuthScheme)
	t.authCredential = cfg.AuthCredential
	t.client = cfg.HTTPClient
	t.headerProvider = cfg.HeaderProvider
	t.blockPrivate = cfg.BlockPrivateNetworks
	return t, nil
}

func newRESTAPITool(name, description string, endpoint Endpoint, operationID string, params []*apiParameter, body *requestBodyInfo, scheme *toolauth.Scheme) *RESTAPITool {
	return &RESTAPITool{
		name:        name,
		description: description,
		endpoint:    endpoint,
		params:      params,
		body:        body,
		authScheme:  scheme,
		parameters:  parametersSchema(operationID, params),
	}
}

// withName returns a copy of t under another name.
func (t *RESTAPITool) withName(name string) *RESTAPITool {
	c := *t
	c.name = name
	return &c
}

// Name implements [tool.Tool].
func (t *RESTAPITool) Name() string {
	return t.name
}

// Description implements [tool.Tool].
func (t *RESTAPITool) Description() string {
	return t.description
}

// IsLongRunning implements [tool.Tool]. It returns false.
func (t *RESTAPITool) IsLongRunning() bool {
	return false
}

// Declaration returns the tool's function declaration. Its parameters are a
// JSON schema with one property per argument, listing the required ones, and
// titled after the operationId: {"properties": ..., "required": [...],
// "title": "<operationId>_Arguments", "type": "object"}.
func (t *RESTAPITool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:                 t.name,
		Description:          t.description,
		ParametersJsonSchema: t.parameters,
	}
}

// ProcessRequest adds the tool's declaration to req.
func (t *RESTAPITool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

// Run sends the HTTP request for the operation and returns its result.
//
// args are the tool-call arguments, keyed by the argument names in the
// declaration. A 2xx JSON object is returned as is; any other 2xx JSON value is
// wrapped as {"result": value}; a non-JSON 2xx body is returned as {"text":
// body}. An HTTP error status, a request that cannot be built, a blocked
// target, or a timeout is reported as {"error": ...} rather than a Go error, so
// the model can read and react to it.
//
// Authentication is not supported yet: an operation whose security requirement
// names a scheme is refused with an error until a later release adds it.
func (t *RESTAPITool) Run(ctx agent.Context, args any) (map[string]any, error) {
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("openapitoolset: unexpected args type %T, want map[string]any", args)
	}
	var extraHeaders map[string]string
	if t.headerProvider != nil {
		headers, err := t.headerProvider(ctx)
		if err != nil {
			return nil, fmt.Errorf("openapitoolset: header provider for tool %q: %w", t.name, err)
		}
		extraHeaders = headers
	}
	return t.run(ctx, m, extraHeaders)
}

// run issues the HTTP request and maps the response. It is the core of Run,
// separated so it needs only a context.Context: Run resolves the header
// provider, which needs the agent context, and passes its headers here.
func (t *RESTAPITool) run(ctx context.Context, args map[string]any, extraHeaders map[string]string) (map[string]any, error) {
	ra, ok, err := t.resolveAuth(ctx)
	if err != nil {
		return nil, fmt.Errorf("openapitoolset: resolve auth for tool %q: %w", t.name, err)
	}
	if !ok {
		// The scheme needs a credential only an interactive flow can obtain,
		// which is not supported yet.
		return t.errorResponse("this operation requires end-user authentication, which is not supported yet"), nil
	}

	prepared, err := t.prepareRequest(args)
	if err != nil {
		var invalid *inputValidationError
		if errors.As(err, &invalid) {
			return t.errorResponse(invalid.Error()), nil
		}
		return nil, fmt.Errorf("openapitoolset: build request for tool %q: %w", t.name, err)
	}
	applyAuth(prepared, ra)

	if t.blockPrivate {
		if err := checkURL(prepared.url); err != nil {
			var invalid *inputValidationError
			if errors.As(err, &invalid) {
				return t.errorResponse(invalid.Error()), nil
			}
			return nil, fmt.Errorf("openapitoolset: validate URL for tool %q: %w", t.name, err)
		}
	}

	req, err := prepared.httpRequest(ctx)
	if err != nil {
		return nil, fmt.Errorf("openapitoolset: build request for tool %q: %w", t.name, err)
	}
	for k, v := range extraHeaders {
		req.Header.Set(k, v)
	}

	client := t.client
	if client == nil {
		client = defaultHTTPClient(t.blockPrivate)
	}
	resp, err := client.Do(req)
	if err != nil {
		return t.errorResponse(requestErrorMessage(err)), nil
	}
	defer func() { _ = resp.Body.Close() }()
	return t.mapResponse(resp)
}

// mapResponse turns an HTTP response into the tool's result.
func (t *RESTAPITool) mapResponse(resp *http.Response) (map[string]any, error) {
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
		return t.errorResponse(fmt.Sprintf("Status Code: %d, %s", resp.StatusCode, string(body))), nil
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openapitoolset: read response for tool %q: %w", t.name, err)
	}
	var decoded any
	if err := json.Unmarshal(body, &decoded); err != nil {
		// A 2xx body that is not JSON is handed back as text, as adk-python does.
		return map[string]any{"text": string(body)}, nil
	}
	if obj, ok := decoded.(map[string]any); ok {
		return obj, nil
	}
	// A non-object JSON value (array, string, number) is wrapped under
	// "result", the shape ADK uses for a tool result that is not a map.
	return map[string]any{"result": decoded}, nil
}

// errorResponse formats a tool-execution error the model can read, with
// adk-python's wording (rest_api_tool.py _format_error_response).
func (t *RESTAPITool) errorResponse(message string) map[string]any {
	return map[string]any{
		"error": fmt.Sprintf(
			"Tool %s execution failed. Analyze this execution error and your inputs."+
				" Retry with adjustments if applicable. But make sure don't retry more"+
				" than 3 times. Execution Error: %s",
			t.name, message,
		),
	}
}

var _ tool.Tool = (*RESTAPITool)(nil)
