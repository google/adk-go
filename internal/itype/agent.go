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

package itype

type AgentConfig struct {
	Name        string
	Description string
	SubAgents   []any // should be *adk.Agent
}

// RegisterNewAgent registers the internal constructor function for adk.Agent.
func RegisterNewAgent(fn func(cfg AgentConfig) (any, error)) {
	configureAgent = fn
}

var configureAgent func(cfg AgentConfig) (any, error)

// NewAgent returns a new adk.Agent.
// The return type is untyped, because this package is imported by the adk package.
func NewAgent(cfg AgentConfig) (any, error) {
	if configureAgent == nil {
		panic("configureAgent is not set")
	}
	return configureAgent(cfg)
}
