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

package shared

import "strings"

// MaxServerTextRunes bounds any server-chosen string an error quotes.
const MaxServerTextRunes = 256

// ClipServerText trims a server-chosen string and caps its length, so that a
// pathological one — a megabyte of message — cannot become the error a caller
// logs. Truncation is marked, so a clipped value does not read as the whole of
// what the server said.
//
// Callers render the result with %q. That, rather than an escaper of our own,
// is what stops a control character in it from forging a line in an operator's
// log; see the same reasoning at [google.golang.org/adk/v2/auth/gcp].
func ClipServerText(s string) string {
	s = strings.TrimSpace(s)
	// Counted by ranging rather than by materialising []rune: an 8 MiB message
	// would otherwise cost 32 MiB to yield at most a kilobyte. Ranging a string
	// yields the byte index of each rune, so s[:i] never splits one.
	n := 0
	for i := range s {
		if n == MaxServerTextRunes {
			return strings.TrimSpace(s[:i]) + "…"
		}
		n++
	}
	return s
}
