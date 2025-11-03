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

package parentmap

import (
	"context"
	"fmt"

	"google.golang.org/adk/agent"
)

type Map map[string]agent.Agent

// New creates parent map allowing to fetch agent's parent.
// It ensures that agent can have at most one parent.
// It ensures that the root node name is not referenced again in the agent tree
func New(root agent.Agent) (Map, error) {
	res := make(map[string]agent.Agent)
	rootName := root.Name()
	pointerMap := map[agent.Agent]string{root: "is root agent"}

	var f func(cur agent.Agent) error
	f = func(cur agent.Agent) error {
		for _, subAgent := range cur.SubAgents() {
			if p, ok := pointerMap[subAgent]; ok {
				return fmt.Errorf("%q agent cannot have >1 parents, found: %q, %q", subAgent.Name(), p, cur.Name())
			}
			if _, ok := res[subAgent.Name()]; ok || subAgent.Name() == rootName {
				return fmt.Errorf("agent names must be unique in the agent tree, found duplicate: %q", subAgent.Name())
			}
			res[subAgent.Name()] = cur
			pointerMap[subAgent] = cur.Name()

			if err := f(subAgent); err != nil {
				return err
			}
		}
		return nil
	}

	return res, f(root)
}

// RootAgent returns the root of the agent tree.
func (m Map) RootAgent(cur agent.Agent) agent.Agent {
	if cur == nil {
		return nil
	}
	for {
		parent := m[cur.Name()]
		if parent == nil {
			return cur
		}
		cur = parent
	}
}

func ToContext(ctx context.Context, parents Map) context.Context {
	return context.WithValue(ctx, mapCtxKey, parents)
}

func FromContext(ctx context.Context) Map {
	m, ok := ctx.Value(mapCtxKey).(Map)
	if !ok {
		return nil
	}
	return m
}

type ctxKey int

const mapCtxKey ctxKey = 0
