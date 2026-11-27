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
	"context"
	"testing"

	"google.golang.org/adk/v2/internal/adkcontext"
)

// panickingValueContext stands in for a context between a reader and the
// runner whose Value panics.
type panickingValueContext struct{ context.Context }

func (panickingValueContext) Value(any) any { panic("broken context") }

func TestExecutionIDFromContext(t *testing.T) {
	runnerCtx := context.WithValue(context.Background(), adkcontext.ExecutionIDKey, "7")
	canceled, cancel := context.WithCancelCause(runnerCtx)
	cancel(nil)
	var unset context.Context
	tests := []struct {
		name   string
		ctx    context.Context
		wantID string
		wantOK bool
	}{
		{name: "set by the runner", ctx: runnerCtx, wantID: "7", wantOK: true},
		{name: "derived and canceled", ctx: canceled, wantID: "7", wantOK: true},
		{name: "nearest runner call wins", ctx: context.WithValue(runnerCtx, adkcontext.ExecutionIDKey, "8"), wantID: "8", wantOK: true},
		{name: "outside a runner call", ctx: context.Background()},
		{name: "nil context", ctx: unset},
		{name: "empty ID", ctx: context.WithValue(context.Background(), adkcontext.ExecutionIDKey, "")},
		{name: "not a string", ctx: context.WithValue(context.Background(), adkcontext.ExecutionIDKey, 7)},
		{name: "other key", ctx: context.WithValue(context.Background(), adkcontext.IdentityKey, "7")},
		{name: "panicking context", ctx: panickingValueContext{runnerCtx}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			id, ok := ExecutionIDFromContext(tc.ctx)
			if id != tc.wantID || ok != tc.wantOK {
				t.Errorf("ExecutionIDFromContext() = (%q, %t), want (%q, %t)", id, ok, tc.wantID, tc.wantOK)
			}
		})
	}
}
