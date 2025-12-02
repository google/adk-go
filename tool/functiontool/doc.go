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

// Package functiontool provides a tool that wraps a Go function.
//
// Special behavior for non-struct inputs:
//
// The LLM `genai.FunctionCall.Args` is always represented as `map[string]any`.
// For Go handlers whose input type (`TArgs`) is a non-struct (for example
// `string`, `int`, or `bool`), ADK's `functiontool.New` wraps the handler with
// a small decorator that:
//  - exposes a function declaration whose `parameters` JSON schema is an
//    `object` with a single required property `"input"` whose schema is the
//    schema inferred for `TArgs`;
//  - at runtime accepts `map[string]any{"input": <value>}` from the model,
//    extracts the `input` value, converts it to `TArgs`, calls the original
//    handler, and converts the result back to `map[string]any` (wrapping
//    primitive outputs as `{ "result": <value> }` when necessary).
//
// This automatic wrapping makes it seamless to author function tools that
// accept primitive arguments while keeping the function-call declaration and
// runtime conversion consistent with the model's map-based function call
// format.
package functiontool
