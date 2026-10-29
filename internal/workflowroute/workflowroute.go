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

// Package workflowroute lets experimental workflow nodes (such as
// exp/classifier.Node) validate their outgoing edges during workflow.New
// without adding experimental hooks to the public workflow API.
package workflowroute

import "sync"

// Kind classifies a concrete route value extracted from a workflow edge.
type Kind uint8

const (
	// KindString represents a string route (workflow.StringRoute or string element in workflow.MultiRoute).
	KindString Kind = iota + 1
	// KindBool represents a boolean route (workflow.BoolRoute or bool element in workflow.MultiRoute).
	KindBool
	// KindOther represents any other concrete route (such as workflow.IntRoute,
	// an empty workflow.MultiRoute, or a custom workflow.Route implementation).
	KindOther
)

// Value describes one concrete route value on an outgoing workflow edge. Label
// is the value in the form a node compares and reports: a quoted string for
// KindString, "true" or "false" for KindBool, and a description otherwise.
type Value struct {
	Kind  Kind
	Label string
}

// OutgoingSummary summarizes all outgoing edges from a single workflow node.
type OutgoingSummary struct {
	HasUnrouted bool
	HasDefault  bool
	Routes      []Value
}

// Validator validates outgoing edges for a node when it recognizes node's concrete type.
type Validator func(node any, out OutgoingSummary) error

var (
	mu         sync.RWMutex
	validators []Validator
)

// Register registers v to run during workflow graph validation. Intended to be
// called from package init functions.
func Register(v Validator) {
	mu.Lock()
	defer mu.Unlock()
	validators = append(validators, v)
}

// Validate runs all registered outgoing-edge validators against node.
func Validate(node any, out OutgoingSummary) error {
	mu.RLock()
	list := validators
	mu.RUnlock()
	for _, v := range list {
		if err := v(node, out); err != nil {
			return err
		}
	}
	return nil
}
