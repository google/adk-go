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

// Package openapitoolset turns an OpenAPI 3 spec into tools: a [Toolset] with
// one [RESTAPITool] per operation, which an agent offers to its model like any
// other tools. It is a port of adk-python's OpenAPIToolset and RestApiTool, and
// names the tools and their arguments the same way, so a spec gives the model
// the same function declarations in both.
//
// This package is experimental; its API and behavior may change or be removed
// in a later version without notice.
//
// Calling a tool sends the HTTP request to the spec's first server URL and
// returns the decoded response. An API key, an HTTP basic or bearer credential,
// an OAuth2 or OpenID Connect credential that already holds an access token, and
// a Google service account are applied directly from [Config.AuthCredential]. A
// scheme that still needs the end user to authorize — an OAuth2 or OpenID
// Connect flow with no token — is not supported yet, and calling such a tool
// returns an error.
//
// A toolset is made from the spec text and given to an agent:
//
//	tools, err := openapitoolset.New(openapitoolset.Config{Spec: specYAML})
//	if err != nil {
//		return err
//	}
//	a, err := llmagent.New(llmagent.Config{
//		Name:     "calendar_agent",
//		Model:    model,
//		Toolsets: []tool.Toolset{tools},
//	})
//
// A single operation can be used without the spec around it, through
// [NewRESTAPITool].
//
// Where this package differs from adk-python:
//
//   - JSON and YAML text are read by one parser, without a format argument.
//     YAML is read as YAML 1.2, where adk-python's yaml.safe_load reads YAML
//     1.1, in which an unquoted on, off, yes or no is a boolean.
//   - [Config.SpecMap] has no key order, so its keys are taken in sorted
//     order, where a Python dict keeps the order it was built in.
//   - The properties of a declaration's parameter schema are a Go map, so
//     they are written in sorted order rather than in the order of the spec.
//   - The prefix of [Config.ToolNamePrefix] is applied by [Toolset.Tools].
//     adk-python applies it in a wrapper around the toolset, which ADK Go
//     does not have.
//   - adk-python logs each tool it makes; this package logs nothing.
package openapitoolset
