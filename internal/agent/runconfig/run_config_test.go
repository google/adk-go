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

package runconfig

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/internal/adkcontext"
)

// snapshotter is the view package agent has of the stored run config: agent
// cannot import this package, so it reads the value under
// adkcontext.RunConfigKey through an interface.
type snapshotter interface {
	CustomMetadataSnapshot() map[string]any
}

func TestCustomMetadataSnapshotThroughSharedKey(t *testing.T) {
	ctx := ToContext(t.Context(), &RunConfig{CustomMetadata: map[string]any{"k": "v"}})

	s, ok := ctx.Value(adkcontext.RunConfigKey).(snapshotter)
	if !ok {
		t.Fatalf("ctx.Value(adkcontext.RunConfigKey) = %T, want a value with CustomMetadataSnapshot", ctx.Value(adkcontext.RunConfigKey))
	}
	if diff := cmp.Diff(map[string]any{"k": "v"}, s.CustomMetadataSnapshot()); diff != "" {
		t.Errorf("CustomMetadataSnapshot() mismatch (-want +got):\n%s", diff)
	}
}

func TestCustomMetadataSnapshotNilRunConfig(t *testing.T) {
	var c *RunConfig
	if got := c.CustomMetadataSnapshot(); got != nil {
		t.Errorf("CustomMetadataSnapshot() on a nil run config = %v, want nil", got)
	}
}
