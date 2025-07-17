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

package base

import (
	"fmt"

	"github.com/google/adk-go"
	"github.com/google/adk-go/internal/itype"
)

type Config = itype.AgentConfig

// NewAgent returns a BaseAgent that can be the base of the implementation agent.
func NewAgent(cfg Config) (*adk.Agent, error) {
	if cfg.Name == "" {
		return nil, fmt.Errorf("agent requires name")
	}
	a, err := itype.NewAgent(cfg)
	if err != nil {
		return nil, err
	}
	if agent, ok := a.(*adk.Agent); ok {
		return agent, nil
	}
	panic(fmt.Sprintf("itype.ConfigureAgent returned unexpected type %T instead of *adk.Agent", a))
}
