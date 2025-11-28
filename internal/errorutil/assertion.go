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

package errorutil

import (
	"errors"
	"testing"
)

// AssertTestError asserts the presence or absence of an error based on expectations.
// It can also check for a specific expected error.
//
// Parameters:
//   - t: the testing instance (marked as Helper for accurate stack traces)
//   - err: the actual error to validate
//   - wantError: boolean indicating if an error is expected
//   - wantSpecificErr: the specific expected error to match against (can be nil if not checking for a specific error)
//   - funcName: descriptive name of the function being tested (for error messages)
//
// Example:
//
//	err := myService.Update(ctx, req)
//	testutil.AssertTestError(t, err, true, session.ErrSessionExpired, "Update()")
func AssertTestError(t *testing.T, err error, wantError bool, wantSpecificErr error, funcName string) {
	t.Helper()

	if !wantError {
		if err != nil {
			t.Fatalf("%s unexpected error: %v", funcName, err)
		}
		return
	}

	if err == nil {
		if wantSpecificErr != nil {
			t.Fatalf("%s expected error %v but got nil", funcName, wantSpecificErr)
		} else {
			t.Fatalf("%s expected an error but got nil", funcName)
		}
		return
	}

	if wantSpecificErr != nil && !errors.Is(err, wantSpecificErr) {
		t.Fatalf("%s error = %v, want %v", funcName, err, wantSpecificErr)
	}
}
