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

package tool_test

import (
	"testing"

	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/agenttool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/tool/geminitool"
	"google.golang.org/adk/tool/loadartifactstool"
)

func TestTypes(t *testing.T) {
	const (
		functionTool = "FunctionTool"
		requestProc  = "RequestProcessor"
	)

	tests := []struct {
		name          string
		constructor   func() (tool.Tool, error)
		expectedTypes []string
	}{
		{
			name: "FunctionTool",
			constructor: func() (tool.Tool, error) {
				return functiontool.New(functiontool.Config{}, func(tool.Context, int) (int, error) { return 0, nil })
			},
			expectedTypes: []string{requestProc, functionTool},
		},
		{
			name:          "geminitool",
			constructor:   func() (tool.Tool, error) { return geminitool.New("", nil), nil },
			expectedTypes: []string{requestProc},
		},
		{
			name:          "geminitool.GoogleSearch{}",
			constructor:   func() (tool.Tool, error) { return geminitool.GoogleSearch{}, nil },
			expectedTypes: []string{requestProc},
		},
		{
			name:          "LoadArtifactsTool",
			constructor:   func() (tool.Tool, error) { return loadartifactstool.New(), nil },
			expectedTypes: []string{requestProc, functionTool},
		},
		{
			name:          "AgentTool",
			constructor:   func() (tool.Tool, error) { return agenttool.New(nil, nil), nil },
			expectedTypes: []string{requestProc, functionTool},
		},
		{
			name:          "LoadArtifactsTool",
			constructor:   func() (tool.Tool, error) { return loadartifactstool.New(), nil },
			expectedTypes: []string{requestProc, functionTool},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool, err := tt.constructor()
			if err != nil {
				t.Fatalf("Failed to create tool %s: %v", tt.name, err)
			}

			for _, s := range tt.expectedTypes {
				switch s {
				case functionTool:
					if _, ok := tool.(toolinternal.FunctionTool); !ok {
						t.Errorf("Expected %s to implement toolinternal.FunctionTool", tt.name)
					}
				case requestProc:
					if _, ok := tool.(toolinternal.RequestProcessor); !ok {
						t.Errorf("Expected %s to implement toolinternal.RequestProcessor", tt.name)
					}
				default:
					t.Fatalf("Unknown expected type: %s", s)
				}
			}
		})
	}
}
