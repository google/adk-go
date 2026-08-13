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

package guardrail_test

import (
	"errors"
	"testing"

	"google.golang.org/adk/v2/guardrail"
)

func TestErrGuardrailBlocked_Error(t *testing.T) {
	tests := []struct {
		name   string
		err    *guardrail.ErrGuardrailBlocked
		want   string
	}{
		{
			name:  "policy and reason",
			err:   &guardrail.ErrGuardrailBlocked{Policy: "pii-filter", Reason: "request contains PII"},
			want:  `guardrail "pii-filter" blocked tool call: request contains PII`,
		},
		{
			name:  "reason only",
			err:   &guardrail.ErrGuardrailBlocked{Reason: "rate limit exceeded"},
			want:  "guardrail blocked tool call: rate limit exceeded",
		},
		{
			name:  "empty",
			err:   &guardrail.ErrGuardrailBlocked{},
			want:  "guardrail blocked tool call",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.err.Error(); got != tt.want {
				t.Errorf("Error() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestErrGuardrailBlocked_ErrorsAs(t *testing.T) {
	original := &guardrail.ErrGuardrailBlocked{Policy: "aml", Reason: "suspicious pattern"}
	wrapped := errors.Join(errors.New("outer"), original)

	var target *guardrail.ErrGuardrailBlocked
	if !errors.As(wrapped, &target) {
		t.Fatal("errors.As should find *ErrGuardrailBlocked inside wrapped error")
	}
	if target.Policy != "aml" {
		t.Errorf("Policy = %q, want %q", target.Policy, "aml")
	}
	if target.Reason != "suspicious pattern" {
		t.Errorf("Reason = %q, want %q", target.Reason, "suspicious pattern")
	}
}

func TestErrGuardrailBlocked_ImplementsError(t *testing.T) {
	var err error = &guardrail.ErrGuardrailBlocked{Policy: "test"}
	if err == nil {
		t.Fatal("*ErrGuardrailBlocked must implement error")
	}
}
