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

// Package adka2a allows exposing ADK agents via A2A.
//
// NewExecutor returns an a2a-go AgentExecutor. To serve it over HTTP, wrap it
// with a2a-go's request handler and transport-specific HTTP handler, such as
// [a2asrv.NewHandler] and [a2asrv.NewJSONRPCHandler].
//
// [a2asrv.NewHandler]: https://pkg.go.dev/github.com/a2aproject/a2a-go/v2/a2asrv#NewHandler
// [a2asrv.NewJSONRPCHandler]: https://pkg.go.dev/github.com/a2aproject/a2a-go/v2/a2asrv#NewJSONRPCHandler
package adka2a
