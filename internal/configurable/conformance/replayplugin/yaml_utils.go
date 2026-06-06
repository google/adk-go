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

package replayplugin

import (
	"strings"

	"gopkg.in/yaml.v3"
)

var toIgnore = map[string]struct{}{
	"thought_signature": {},
	"http_options":      {},
	"args":              {},
	"response":          {},
}

var toIgnoreButRecurse = map[string]struct{}{
	"user_message_index": {},
	"agent_name":         {},
	"llm_recording":      {},
	"llm_request":        {},
	"llm_responses":      {},
	"tool_recording":     {},
	"tool_call":          {},
	"tool_response":      {},
}

// normalizeYAMLNode standardizes the recorded YAML AST in a single tree traversal.
// It strips underscores from structural Go field keys to match case-insensitive fields,
// while preserving user freeform tool arguments/responses, and resolves type mismatches.
func normalizeYAMLNode(node *yaml.Node) {
	switch node.Kind {
	case yaml.DocumentNode, yaml.SequenceNode:
		for _, child := range node.Content {
			normalizeYAMLNode(child)
		}
	case yaml.MappingNode:
		for i := 0; i < len(node.Content); i += 2 {
			keyNode := node.Content[i]
			valueNode := node.Content[i+1]

			// 1. Fix known type mismatches and singular/plural fields
			switch keyNode.Value {
			case "systeminstruction":
				if valueNode.Kind == yaml.ScalarNode {
					val := valueNode.Value
					valueNode.Kind = yaml.MappingNode
					valueNode.Tag = "!!map"
					valueNode.Value = ""
					valueNode.Content = []*yaml.Node{
						{Kind: yaml.ScalarNode, Value: "role"},
						{Kind: yaml.ScalarNode, Tag: "!!str", Value: "user"},
						{Kind: yaml.ScalarNode, Value: "parts"},
						{
							Kind: yaml.SequenceNode,
							Tag:  "!!seq",
							Content: []*yaml.Node{
								{
									Kind: yaml.MappingNode,
									Tag:  "!!map",
									Content: []*yaml.Node{
										{Kind: yaml.ScalarNode, Value: "text"},
										{Kind: yaml.ScalarNode, Tag: "!!str", Value: val},
									},
								},
							},
						},
					}
				}
			case "llmresponse", "llm_response":
				if valueNode.Kind == yaml.MappingNode {
					origCopy := *valueNode
					valueNode.Kind = yaml.SequenceNode
					valueNode.Tag = "!!seq"
					valueNode.Value = ""
					valueNode.Content = []*yaml.Node{&origCopy}
				}
				keyNode.Value = "llm_responses"
			}

			// 2. Skip processing/recursion for ignored freeform keys (like tool args or response payload)
			if _, ok := toIgnore[keyNode.Value]; ok {
				continue
			}

			// 3. For standard metadata fields, preserve snake_case but recurse into value
			if _, ok := toIgnoreButRecurse[keyNode.Value]; ok {
				normalizeYAMLNode(valueNode)
				continue
			}

			// 4. Strip underscores to match camelCase Go struct field names case-insensitively
			keyNode.Value = strings.ReplaceAll(keyNode.Value, "_", "")

			// 5. Recurse to process nested fields
			normalizeYAMLNode(valueNode)
		}
	}
}
