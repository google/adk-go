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

// BaseNode provides identity and a default Config implementation for
// types that satisfy the Node interface. Custom node types embed
// BaseNode by value and supply only Run.
type BaseNode struct {
	name   string
	desc   string
	config NodeConfig
}

// NewBaseNode returns a BaseNode with the given identity and
// configuration. Embedders typically call it from their own
// constructor:
//
//	type CustomNode struct {
//	    BaseNode
//	    // ...
//	}
//
//	func NewCustomNode(name string, cfg NodeConfig) *CustomNode {
//	    return &CustomNode{BaseNode: NewBaseNode(name, "", cfg)}
//	}
func NewBaseNode(name, description string, cfg NodeConfig) BaseNode {
	return BaseNode{name: name, desc: description, config: cfg}
}

// Name returns the node's name.
func (b BaseNode) Name() string { return b.name }

// Description returns the node's human-readable description.
func (b BaseNode) Description() string { return b.desc }

// Config returns the node's configuration.
func (b BaseNode) Config() NodeConfig { return b.config }
