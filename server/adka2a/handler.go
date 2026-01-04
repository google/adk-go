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

package adka2a

import "github.com/a2aproject/a2a-go/a2asrv"

// HandlerConfig allows to configure the A2A request handler.
type HandlerConfig struct {
	// ExecutorConfig is the configuration for the A2A executor.
	ExecutorConfig ExecutorConfig

	// A2AOptions are optional configurations for the A2A request handler.
	A2AOptions []a2asrv.RequestHandlerOption
}

// NewRequestHandler creates a transport-agnostic A2A request handler.
// Callers can wrap the returned handler with a transport implementation such as
// a2asrv.NewJSONRPCHandler.
func NewRequestHandler(config HandlerConfig) a2asrv.RequestHandler {
	executor := NewExecutor(config.ExecutorConfig)
	return a2asrv.NewHandler(executor, config.A2AOptions...)
}
