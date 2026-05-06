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
)

func TestApplyDefaults(t *testing.T) {
	tests := []struct {
		name              string
		input             *RetryConfig
		wantMaxAttempts   int
		wantInitialDelay  time.Duration
		wantMaxDelay      time.Duration
		wantBackoffFactor float64
		wantJitter        float64
	}{
		{
			name:              "all_fields_nil",
			input:             &RetryConfig{},
			wantMaxAttempts:   5,
			wantInitialDelay:  1 * time.Second,
			wantMaxDelay:      60 * time.Second,
			wantBackoffFactor: 2.0,
			wantJitter:        1.0,
		},
		{
			name: "some_fields_set",
			input: &RetryConfig{
				MaxAttempts: ptr(10),
				Jitter:      ptr(0.0),
			},
			wantMaxAttempts:   10,
			wantInitialDelay:  1 * time.Second,
			wantMaxDelay:      60 * time.Second,
			wantBackoffFactor: 2.0,
			wantJitter:        0.0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.input.applyDefaults()

			if *tc.input.MaxAttempts != tc.wantMaxAttempts {
				t.Errorf("MaxAttempts = %d, want %d", *tc.input.MaxAttempts, tc.wantMaxAttempts)
			}
			if *tc.input.InitialDelay != tc.wantInitialDelay {
				t.Errorf("InitialDelay = %v, want %v", *tc.input.InitialDelay, tc.wantInitialDelay)
			}
			if *tc.input.MaxDelay != tc.wantMaxDelay {
				t.Errorf("MaxDelay = %v, want %v", *tc.input.MaxDelay, tc.wantMaxDelay)
			}
			if *tc.input.BackoffFactor != tc.wantBackoffFactor {
				t.Errorf("BackoffFactor = %f, want %f", *tc.input.BackoffFactor, tc.wantBackoffFactor)
			}
			if *tc.input.Jitter != tc.wantJitter {
				t.Errorf("Jitter = %f, want %f", *tc.input.Jitter, tc.wantJitter)
			}
			if tc.input.ShouldRetry == nil || !tc.input.ShouldRetry(nil) {
				t.Errorf("ShouldRetry is nil or returns false")
			}
		})
	}
}

