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

// Package pathquote percent-encodes paths the way the artifact services of
// adk-python do, so that the URLs and URIs the Go services build match
// Python's byte for byte.
package pathquote

import "strings"

// Quote percent-encodes s as adk-python's urllib.parse.quote(s, safe="/"):
// ASCII letters and digits and "-", ".", "_", "~" and "/" stay as they are,
// and every other byte of s becomes "%XX" with upper-case hex digits.
// url.PathEscape differs: it keeps characters such as ":" and "@".
func Quote(s string) string {
	const hexDigits = "0123456789ABCDEF"
	var b strings.Builder
	for i := range len(s) {
		c := s[i]
		if 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || '0' <= c && c <= '9' || strings.IndexByte("-._~/", c) >= 0 {
			b.WriteByte(c)
			continue
		}
		b.WriteByte('%')
		b.WriteByte(hexDigits[c>>4])
		b.WriteByte(hexDigits[c&0x0f])
	}
	return b.String()
}
