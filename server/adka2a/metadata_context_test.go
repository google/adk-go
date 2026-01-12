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

package adka2a

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestA2AMetadataContext(t *testing.T) {
	testCases := []struct {
		name string
		meta *A2AMetadata
	}{
		{
			name: "full metadata",
			meta: &A2AMetadata{
				TaskID:          "task-123",
				ContextID:       "ctx-456",
				RequestMetadata: map[string]any{"trace_id": "trace-789"},
				MessageMetadata: map[string]any{"correlation_id": "corr-abc"},
			},
		},
		{
			name: "task id only",
			meta: &A2AMetadata{
				TaskID: "task-123",
			},
		},
		{
			name: "empty metadata",
			meta: &A2AMetadata{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()

			// Initially no metadata
			if got := A2AMetadataFromContext(ctx); got != nil {
				t.Errorf("A2AMetadataFromContext() = %v, want nil", got)
			}

			// Add metadata
			ctx = ContextWithA2AMetadata(ctx, tc.meta)

			// Should retrieve the same metadata
			got := A2AMetadataFromContext(ctx)
			if diff := cmp.Diff(tc.meta, got); diff != "" {
				t.Errorf("A2AMetadataFromContext() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestA2AMetadataFromContext_NoMetadata(t *testing.T) {
	ctx := context.Background()
	got := A2AMetadataFromContext(ctx)
	if got != nil {
		t.Errorf("A2AMetadataFromContext() = %v, want nil", got)
	}
}
