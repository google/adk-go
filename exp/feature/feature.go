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

// Package feature holds the switches of ADK Go features that are not yet
// stable. A switch travels in a context.Context, so one process can run
// agents with different values at the same time.
//
// EXPERIMENTAL: This package is experimental; its API and behavior may change or be removed in a later version.
package feature

import "context"

type legacyKey struct{}

// WithLegacyFunctionDeclarationSchema returns a copy of parent with the legacy
// function-declaration switch set to v. When it is true, function tools
// (functiontool.New and functiontool.NewStreaming), the set_model_response
// tool and the transfer_to_agent tool (the built-in one and the one
// transfertoagenttool.New builds) declare their parameters in the typed
// genai.Schema form (Parameters) instead of a JSON Schema
// (ParametersJsonSchema). Safe for concurrent use.
//
// EXPERIMENTAL: WithLegacyFunctionDeclarationSchema is experimental; its API and behavior may change or be removed in a later version.
func WithLegacyFunctionDeclarationSchema(parent context.Context, v bool) context.Context {
	return context.WithValue(parent, legacyKey{}, v)
}

// LegacyFunctionDeclarationSchema returns the value of the switch in ctx, or
// false if it is unset. Safe for concurrent use.
//
// EXPERIMENTAL: LegacyFunctionDeclarationSchema is experimental; its API and behavior may change or be removed in a later version.
func LegacyFunctionDeclarationSchema(ctx context.Context) bool {
	v, _ := ctx.Value(legacyKey{}).(bool)
	return v
}
