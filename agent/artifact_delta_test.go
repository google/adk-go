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

package agent

import (
	"errors"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/session"
)

func TestMergeArtifactDelta_SupportedContexts(t *testing.T) {
	builders := []struct {
		name  string
		build func(ic InvocationContext, actions *session.EventActions) Context
	}{
		{name: "NewCallbackContext", build: NewCallbackContext},
		{name: "NewCallbackContextWithArtifactTracking", build: NewCallbackContextWithArtifactTracking},
		{
			name: "NewToolContext",
			build: func(ic InvocationContext, actions *session.EventActions) Context {
				return NewToolContext(ic, "call-1", actions, nil)
			},
		},
	}
	derivations := []struct {
		name   string
		derive func(t *testing.T, ctx Context) Context
	}{
		{name: "as built", derive: func(t *testing.T, ctx Context) Context { return ctx }},
		{name: "WithDelta", derive: func(t *testing.T, ctx Context) Context { return ctx.WithDelta(&CommonContextDelta{}) }},
		{
			name: "WithAgentCancel",
			derive: func(t *testing.T, ctx Context) Context {
				derived, cancel := ctx.WithAgentCancel()
				t.Cleanup(cancel)
				return derived
			},
		},
		{
			name: "WithAgentTimeout",
			derive: func(t *testing.T, ctx Context) Context {
				derived, cancel := ctx.WithAgentTimeout(time.Minute)
				t.Cleanup(cancel)
				return derived
			},
		},
	}
	for _, b := range builders {
		for _, d := range derivations {
			t.Run(b.name+"/"+d.name, func(t *testing.T) {
				actions := &session.EventActions{}
				ctx := d.derive(t, b.build(&invocationContext{Context: t.Context()}, actions))

				first := map[string]int64{"a.pdf": 2, "b.pdf": 1}
				if err := MergeArtifactDelta(ctx, first); err != nil {
					t.Fatalf("MergeArtifactDelta(first) error = %v, want nil", err)
				}
				if diff := cmp.Diff(map[string]int64{"a.pdf": 2, "b.pdf": 1}, actions.ArtifactDelta); diff != "" {
					t.Fatalf("ArtifactDelta after first merge mismatch (-want +got):\n%s", diff)
				}

				// A lower version replaces a higher one: the delta is merged as
				// adk-python's dict |= does, not kept at the maximum.
				if err := MergeArtifactDelta(ctx, map[string]int64{"a.pdf": 1, "c.pdf": 4}); err != nil {
					t.Fatalf("MergeArtifactDelta(second) error = %v, want nil", err)
				}
				want := map[string]int64{"a.pdf": 1, "b.pdf": 1, "c.pdf": 4}
				if diff := cmp.Diff(want, actions.ArtifactDelta); diff != "" {
					t.Fatalf("ArtifactDelta after second merge mismatch (-want +got):\n%s", diff)
				}

				if err := MergeArtifactDelta(ctx, nil); err != nil {
					t.Errorf("MergeArtifactDelta(nil) error = %v, want nil", err)
				}
				if err := MergeArtifactDelta(ctx, map[string]int64{}); err != nil {
					t.Errorf("MergeArtifactDelta(empty) error = %v, want nil", err)
				}
				first["a.pdf"] = 9
				first["d.pdf"] = 5
				if diff := cmp.Diff(want, actions.ArtifactDelta); diff != "" {
					t.Errorf("ArtifactDelta after empty merges and a change to the caller's map mismatch (-want +got):\n%s", diff)
				}
			})
		}
	}
}

func TestMergeArtifactDelta_UnsupportedContexts(t *testing.T) {
	contexts := []struct {
		name  string
		build func(t *testing.T) Context
	}{
		{
			name:  "NewContext",
			build: func(t *testing.T) Context { return NewContext(&invocationContext{Context: t.Context()}) },
		},
		{
			name:  "Promote",
			build: func(t *testing.T) Context { return Promote(&invocationContext{Context: t.Context()}) },
		},
		{
			name:  "StrictContextMock",
			build: func(t *testing.T) Context { return &StrictContextMock{Ctx: t.Context()} },
		},
	}
	deltas := []struct {
		name  string
		delta map[string]int64
	}{
		{name: "non-empty", delta: map[string]int64{"a.pdf": 1}},
		{name: "nil", delta: nil},
		{name: "empty", delta: map[string]int64{}},
	}
	for _, c := range contexts {
		for _, d := range deltas {
			t.Run(c.name+"/"+d.name, func(t *testing.T) {
				err := MergeArtifactDelta(c.build(t), d.delta)
				if !errors.Is(err, errors.ErrUnsupported) {
					t.Errorf("MergeArtifactDelta() error = %v, want an error wrapping errors.ErrUnsupported", err)
				}
			})
		}
	}
}
