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

package flowtool

import (
	"encoding/json"
	"fmt"
)

// SpecKind enumerates flow node kinds.
type SpecKind string

const (
	KindAgent    SpecKind = "agent"
	KindSeq      SpecKind = "seq"
	KindParallel SpecKind = "parallel"
)

// Spec is the recursive flow specification an LLM emits to run_flow.
//
// Exactly one of Agent / Seq / Parallel is set, matching Type.
type Spec struct {
	Type SpecKind `json:"type"`

	// Agent fields (Type == KindAgent).
	Agent string `json:"agent,omitempty"`
	Input string `json:"input,omitempty"`

	// Seq / Parallel fields.
	Nodes []Spec `json:"nodes,omitempty"`

	// Path is set by AssignPaths. It is the dotted address of this node
	// in the spec tree, e.g. "seq[0].researcher" or
	// "seq[1].parallel[0].drafter".
	Path string `json:"-"`
}

// UnmarshalJSON enforces the discriminated-union shape and rejects unknown
// kinds early.
func (s *Spec) UnmarshalJSON(data []byte) error {
	type raw struct {
		Type  SpecKind `json:"type"`
		Agent string   `json:"agent,omitempty"`
		Input string   `json:"input,omitempty"`
		Nodes []Spec   `json:"nodes,omitempty"`
	}
	var r raw
	if err := json.Unmarshal(data, &r); err != nil {
		return err
	}
	switch r.Type {
	case KindAgent:
		if r.Agent == "" {
			return fmt.Errorf("flowtool: agent node missing %q field", "agent")
		}
		if len(r.Nodes) > 0 {
			return fmt.Errorf("flowtool: agent node %q must not have nodes", r.Agent)
		}
	case KindSeq, KindParallel:
		if len(r.Nodes) == 0 {
			return fmt.Errorf("flowtool: %s node has empty nodes list", r.Type)
		}
		if r.Agent != "" || r.Input != "" {
			return fmt.Errorf("flowtool: %s node must not set agent/input", r.Type)
		}
	default:
		return fmt.Errorf("flowtool: unknown spec type %q (want agent|seq|parallel)", r.Type)
	}
	*s = Spec{
		Type:  r.Type,
		Agent: r.Agent,
		Input: r.Input,
		Nodes: r.Nodes,
	}
	return nil
}

// AssignPaths walks the spec tree and fills Spec.Path on every node.
// The root node receives the supplied prefix (typically empty).
func AssignPaths(s *Spec, prefix string) {
	switch s.Type {
	case KindAgent:
		if prefix == "" {
			s.Path = s.Agent
		} else {
			s.Path = prefix + "." + s.Agent
		}
	case KindSeq:
		s.Path = joinPath(prefix, "seq")
		for i := range s.Nodes {
			child := &s.Nodes[i]
			AssignPaths(child, fmt.Sprintf("%s[%d]", s.Path, i))
		}
	case KindParallel:
		s.Path = joinPath(prefix, "parallel")
		for i := range s.Nodes {
			child := &s.Nodes[i]
			AssignPaths(child, fmt.Sprintf("%s[%d]", s.Path, i))
		}
	}
}

func joinPath(prefix, segment string) string {
	if prefix == "" {
		return segment
	}
	return prefix + "." + segment
}

// Walk visits every node in the spec tree in pre-order.
func Walk(s *Spec, fn func(*Spec)) {
	fn(s)
	for i := range s.Nodes {
		Walk(&s.Nodes[i], fn)
	}
}

// Depth returns the maximum nesting depth of the spec tree. A leaf agent is
// depth 1; seq/parallel adds one level.
func Depth(s *Spec) int {
	if s.Type == KindAgent {
		return 1
	}
	max := 0
	for i := range s.Nodes {
		d := Depth(&s.Nodes[i])
		if d > max {
			max = d
		}
	}
	return 1 + max
}

// CountNodes returns the total number of agent leaves plus seq/parallel
// nodes in the tree.
func CountNodes(s *Spec) int {
	n := 1
	for i := range s.Nodes {
		n += CountNodes(&s.Nodes[i])
	}
	return n
}

// MaxParallelWidth returns the largest fan-out width of any parallel node
// in the tree, or 0 if none.
func MaxParallelWidth(s *Spec) int {
	max := 0
	if s.Type == KindParallel {
		max = len(s.Nodes)
	}
	for i := range s.Nodes {
		if w := MaxParallelWidth(&s.Nodes[i]); w > max {
			max = w
		}
	}
	return max
}
