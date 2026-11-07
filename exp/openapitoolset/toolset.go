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
	"errors"
	"fmt"
	"net/http"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolauth"
)

// Config configures a [Toolset].
type Config struct {
	// Spec is the OpenAPI 3 spec as JSON or YAML text. Set exactly one of
	// Spec and SpecMap.
	Spec []byte
	// SpecMap is the OpenAPI 3 spec as decoded JSON. A Go map has no order,
	// so its keys are taken in sorted order. With Spec, the order of the text
	// decides the order of the tools and of each tool's arguments.
	SpecMap map[string]any

	// PreservePropertyNames keeps the spec's names for the tools' arguments.
	// By default they are converted to snake_case, so calendarId becomes
	// calendar_id. Either way, a name that is a Python keyword, such as "in",
	// gets the prefix "param_", as adk-python names it.
	PreservePropertyNames bool

	// ToolNamePrefix, when set, is put in front of every tool name that
	// Toolset.Tools returns, followed by "_", so that toolsets for APIs with
	// similar operations can serve one agent. ToolFilter and Toolset.Tool see
	// the names without it.
	ToolNamePrefix string

	// ToolFilter, when set, selects the tools Toolset.Tools returns. It is
	// called with each tool under its unprefixed name.
	ToolFilter tool.Predicate

	// AuthScheme, when set, replaces the security scheme every tool took from
	// the spec, so one scheme authenticates them all. A tool that had no scheme
	// gets this one too.
	AuthScheme *toolauth.Scheme
	// AuthCredential, when set, is the credential every tool authenticates
	// with. An API key, an HTTP basic or bearer credential, an OAuth2 or OpenID
	// Connect credential that already holds an access token, and a Google
	// service account are applied directly. A scheme that still needs the end
	// user to authorize (an OAuth2 or OpenID Connect flow with no token) is not
	// supported yet: calling such a tool returns an error.
	AuthCredential *toolauth.Credential

	// HTTPClient sends every tool's request. When nil, each tool uses a client
	// with bounded timeouts (a short connect timeout and a generous overall
	// one). Supply one to set a proxy, a custom TLS config, or any other
	// transport option. A supplied client is used as given, so BlockPrivateNetworks
	// then applies only the lexical scheme and hostname checks, not the dial-time
	// IP guard; the client is responsible for its own egress policy.
	HTTPClient *http.Client

	// HeaderProvider, when set, is called for each request and its headers are
	// set on the request, overriding any the tool set. Use it for request
	// metadata such as a correlation ID.
	HeaderProvider func(agent.ReadonlyContext) (map[string]string, error)

	// BlockPrivateNetworks refuses requests whose target is a loopback,
	// private, link-local (including the cloud metadata endpoint) or
	// "*.internal" / "*.local" address, and non-HTTP schemes. It is off by
	// default. With the default client the check runs against the address
	// actually dialed, so it also covers a hostname that resolves to such an
	// address; with a supplied HTTPClient only the lexical checks apply.
	BlockPrivateNetworks bool
}

// Toolset is a [tool.Toolset] with one [RESTAPITool] for each operation of an
// OpenAPI spec. Its tools are made by [New] and do not change, and it is safe
// for concurrent use.
type Toolset struct {
	// tools are the tools under their own names, and listed are the same
	// tools as Tools returns them, with the prefix.
	tools  []*RESTAPITool
	listed []tool.Tool
	filter tool.Predicate
}

// New parses the spec in cfg into one tool per operation. It fails when the
// spec is not a JSON or YAML object, uses a reference to another document, or
// has an operation that cannot be parsed.
//
// Each method of each path becomes a tool named after the operation's
// operationId in snake_case, or after its path and method when it has none,
// so "calendar.calendars.get" becomes calendar_calendars_get and GET /test
// becomes test_get. Names are cut to 60 characters, and a name that is
// already taken gets the first free suffix _2, _3 and so on. A tool's
// description is the operation's description, else its summary.
//
// A tool's arguments are the operation's path, query, header and cookie
// parameters, including those its path declares, followed by its request
// body: the properties of an object body, one argument named "array" for an
// array body, and one named "body" for any other. A name that clashes with an
// earlier one gets a suffix _0, _1 and so on. Local $ref references are
// resolved first, and schema types that JSON Schema does not have, such as
// "Any", are dropped.
//
// The tools send their requests to the first server URL in the spec.
func New(cfg Config) (*Toolset, error) {
	spec, err := cfg.loadSpec()
	if err != nil {
		return nil, err
	}
	ops, err := parseSpec(spec, cfg.PreservePropertyNames)
	if err != nil {
		return nil, fmt.Errorf("openapitoolset: %w", err)
	}
	tools := make([]*RESTAPITool, len(ops))
	for i, op := range ops {
		name := truncate(snakeCase(op.name), maxToolNameLength)
		scheme := op.authScheme
		if cfg.AuthScheme != nil {
			scheme = cfg.AuthScheme
		}
		t := newRESTAPITool(name, op.description, op.endpoint, op.operationID, op.params, op.body, scheme)
		t.authCredential = cfg.AuthCredential
		t.client = cfg.HTTPClient
		t.headerProvider = cfg.HeaderProvider
		t.blockPrivate = cfg.BlockPrivateNetworks
		tools[i] = t
	}
	disambiguateToolNames(tools)

	listed := make([]tool.Tool, len(tools))
	for i, t := range tools {
		listed[i] = t
		if cfg.ToolNamePrefix != "" {
			listed[i] = t.withName(cfg.ToolNamePrefix + "_" + t.name)
		}
	}
	return &Toolset{tools: tools, listed: listed, filter: cfg.ToolFilter}, nil
}

func (cfg *Config) loadSpec() (*object, error) {
	switch {
	case len(cfg.Spec) > 0 && len(cfg.SpecMap) > 0:
		return nil, errors.New("openapitoolset: set only one of Config.Spec and Config.SpecMap")
	case len(cfg.SpecMap) > 0:
		spec, err := objectFromMap(cfg.SpecMap)
		if err != nil {
			return nil, fmt.Errorf("openapitoolset: Config.SpecMap: %w", err)
		}
		return spec, nil
	case len(cfg.Spec) > 0:
		v, err := decodeDocument(cfg.Spec)
		if err != nil {
			return nil, fmt.Errorf("openapitoolset: Config.Spec: %w", err)
		}
		spec, ok := v.(*object)
		if !ok {
			return nil, errors.New("openapitoolset: the OpenAPI spec must be an object")
		}
		return spec, nil
	}
	return nil, errors.New("openapitoolset: set Config.Spec or Config.SpecMap")
}

// disambiguateToolNames renames, in place, each tool whose name an earlier
// tool already has, to the name with the first suffix _2, _3, ... that no
// tool has, cut so that it stays within maxToolNameLength. Operations whose
// IDs share a long prefix otherwise end up with one name. It is adk-python's
// _disambiguate_tool_names.
func disambiguateToolNames(tools []*RESTAPITool) {
	reserved := make(map[string]bool, len(tools))
	for _, t := range tools {
		reserved[t.name] = true
	}
	assigned := make(map[string]bool, len(tools))
	for _, t := range tools {
		if assigned[t.name] {
			base := t.name
			for n := 2; assigned[t.name] || reserved[t.name]; n++ {
				suffix := fmt.Sprintf("_%d", n)
				t.name = truncate(base, maxToolNameLength-len(suffix)) + suffix
			}
		}
		assigned[t.name] = true
	}
}

// Name implements [tool.Toolset].
func (*Toolset) Name() string {
	return "openapi_toolset"
}

// Tools implements [tool.Toolset]. It returns the tools Config.ToolFilter
// selects, in the order of the spec, with Config.ToolNamePrefix applied.
func (s *Toolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	out := make([]tool.Tool, 0, len(s.tools))
	for i, t := range s.tools {
		if s.filter != nil && !s.filter(ctx, t) {
			continue
		}
		out = append(out, s.listed[i])
	}
	return out, nil
}

// Tool returns the first tool named name, looked up by its name without
// Config.ToolNamePrefix and whether or not Config.ToolFilter selects it.
func (s *Toolset) Tool(name string) (*RESTAPITool, bool) {
	for _, t := range s.tools {
		if t.name == name {
			return t, true
		}
	}
	return nil, false
}

var _ tool.Toolset = (*Toolset)(nil)
