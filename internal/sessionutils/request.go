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

package sessionutils

import "fmt"

// CheckNumRecentEvents returns the error every session.Service returns for a
// negative GetRequest.NumRecentEvents, and nil for zero or a positive n. The
// error is a plain validation error: it never wraps session.ErrNotFound.
func CheckNumRecentEvents(n int) error {
	if n < 0 {
		return fmt.Errorf("num_recent_events must not be negative, got %d", n)
	}
	return nil
}
