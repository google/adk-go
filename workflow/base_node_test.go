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
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
)

// Compile-time assertions: every built-in workflow node must satisfy
// the Node interface.
var (
	_ Node = (*startNode)(nil)
	_ Node = (*FunctionNode)(nil)
)

func TestNewBaseNode_RoundTrip(t *testing.T) {
	tTrue := true
	tFalse := false
	tests := []struct {
		name       string
		nameArg    string
		descArg    string
		cfg        NodeConfig
		wantConfig NodeConfig
	}{
		{
			name:    "zero config",
			nameArg: "n",
			descArg: "desc",
		},
		{
			name:       "WaitForOutput=true (JoinNode shape)",
			nameArg:    "join",
			descArg:    "fan-in",
			cfg:        NodeConfig{WaitForOutput: &tTrue},
			wantConfig: NodeConfig{WaitForOutput: &tTrue},
		},
		{
			name:       "ParallelWorker=true",
			nameArg:    "mapper",
			descArg:    "data parallel",
			cfg:        NodeConfig{ParallelWorker: true},
			wantConfig: NodeConfig{ParallelWorker: true},
		},
		{
			name:       "empty name and description",
			cfg:        NodeConfig{},
			wantConfig: NodeConfig{},
		},
		{
			name:    "fully populated configuration",
			nameArg: "full_node",
			descArg: "Node with all config fields set",
			cfg: NodeConfig{
				ParallelWorker: true,
				RerunOnResume:  &tFalse,
				WaitForOutput:  &tTrue,
				RetryConfig: &RetryConfig{
					MaxAttempts: 3,
				},
				Timeout: 10 * time.Second,
			},
			wantConfig: NodeConfig{
				ParallelWorker: true,
				RerunOnResume:  &tFalse,
				WaitForOutput:  &tTrue,
				RetryConfig: &RetryConfig{
					MaxAttempts: 3,
				},
				Timeout: 10 * time.Second,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := NewBaseNode(tt.nameArg, tt.descArg, tt.cfg)
			if got := b.Name(); got != tt.nameArg {
				t.Errorf("Name() = %q, want %q", got, tt.nameArg)
			}
			if got := b.Description(); got != tt.descArg {
				t.Errorf("Description() = %q, want %q", got, tt.descArg)
			}
			want := tt.wantConfig
			if (tt.cfg.WaitForOutput == nil) && (tt.cfg.ParallelWorker == false) {
				want = NodeConfig{}
			}
			if diff := cmp.Diff(want, b.Config()); diff != "" {
				t.Errorf("Config() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
