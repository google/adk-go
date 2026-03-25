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

var toIgnore = map[string]struct{}{"thought_signature": {}, "http_options": {}, "args": {}, "response": {}}

func removeUnderscores(node *yaml.Node) {
	switch node.Kind {
	case yaml.DocumentNode, yaml.SequenceNode:
		// If it's a document or a list, just pass through to its children
		for _, child := range node.Content {
			removeUnderscores(child)
		}
	case yaml.MappingNode:
		// A MappingNode's content is a flat array alternating [key, value, key, value...]
		for i := 0; i < len(node.Content); i += 2 {
			keyNode := node.Content[i]
			valueNode := node.Content[i+1]
			if _, ok := toIgnore[keyNode.Value]; ok {
				continue
			}

			// Strip the underscore from the key (e.g., "first_name" -> "firstname")
			keyNode.Value = strings.ReplaceAll(keyNode.Value, "_", "")

			// Continue walking down into the value in case of nested objects
			removeUnderscores(valueNode)
		}
	}
}

func fixTypeMismatches(n *yaml.Node) {
	switch n.Kind {
	case yaml.DocumentNode, yaml.SequenceNode:
		for _, child := range n.Content {
			fixTypeMismatches(child)
		}

	case yaml.MappingNode:
		for i := 0; i < len(n.Content); i += 2 {
			keyNode := n.Content[i]
			valueNode := n.Content[i+1]

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
			}

			// Recurse into the value to catch nested structures
			fixTypeMismatches(valueNode)
		}
	}
}
