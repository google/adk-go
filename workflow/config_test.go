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
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestApplyDefaults(t *testing.T) {
	tests := []struct {
		name  string
		input *RetryConfig
		want  *RetryConfig
	}{
		{
			name:  "all_fields_nil",
			input: &RetryConfig{},
			want: &RetryConfig{
				MaxAttempts:   ptr(5),
				InitialDelay:  ptr(1 * time.Second),
				MaxDelay:      ptr(60 * time.Second),
				BackoffFactor: ptr(2.0),
				Jitter:        ptr(1.0),
			},
		},
		{
			name: "some_fields_set",
			input: &RetryConfig{
				MaxAttempts: ptr(10),
				Jitter:      ptr(0.0),
			},
			want: &RetryConfig{
				MaxAttempts:   ptr(10),
				InitialDelay:  ptr(1 * time.Second),
				MaxDelay:      ptr(60 * time.Second),
				BackoffFactor: ptr(2.0),
				Jitter:        ptr(0.0),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.input.applyDefaults()

			if diff := cmp.Diff(tc.want, tc.input, cmpopts.IgnoreFields(RetryConfig{}, "ShouldRetry")); diff != "" {
				t.Errorf("applyDefaults() mismatch (-want +got):\n%s", diff)
			}

			// Check ShouldRetry separately
			if tc.input.ShouldRetry == nil || !tc.input.ShouldRetry(nil) {
				t.Errorf("ShouldRetry is nil or returns false")
			}
		})
	}
}


