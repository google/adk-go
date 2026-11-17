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

package workflow

import (
	"iter"
	"slices"
	"sync"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// faninNode is a static node whose body each test supplies.
type faninNode struct {
	BaseNode
	run func(ctx agent.Context, input any, yield func(*session.Event, error) bool)
}

func newFaninNode(name string, cfg NodeConfig, run func(ctx agent.Context, input any, yield func(*session.Event, error) bool)) *faninNode {
	return &faninNode{BaseNode: NewBaseNode(name, "", cfg), run: run}
}

func (n *faninNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.run(ctx, input, yield)
	}
}

// faninObs is one observation of a node activation.
type faninObs struct {
	Path   string
	Branch string
	Value  any
}

// faninLog collects observations from node goroutines.
type faninLog struct {
	mu    sync.Mutex
	items []faninObs
}

func (l *faninLog) add(o faninObs) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.items = append(l.items, o)
}

func (l *faninLog) get() []faninObs {
	l.mu.Lock()
	defer l.mu.Unlock()
	return slices.Clone(l.items)
}
