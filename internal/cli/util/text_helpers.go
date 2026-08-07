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

package util

import (
	"fmt"
	"regexp"
	"strings"
)

func CenterString(s string, w int) string {
	sw := w - len(s)
	lw := sw / 2
	rw := sw - lw
	return strings.Repeat(" ", lw) + s + strings.Repeat(" ", rw)
}

// ValidateDockerfileSafe reports an error if s contains a character that
// would let it break out of the double-quoted string, or out of the current
// instruction entirely, when embedded literally into generated Dockerfile
// content (Dockerfile instructions are newline-delimited, and none of them
// define a generic escaping mechanism for interpolated values).
func ValidateDockerfileSafe(s, label string) error {
	if strings.ContainsAny(s, "\"`\n\r\x00") {
		return fmt.Errorf("%s %q contains a character (quote, backtick, newline, or NUL) that is not allowed in a value embedded in generated Dockerfile content", label, s)
	}
	return nil
}

var shellArgSafeRE = regexp.MustCompile(`^[A-Za-z0-9_./-]*$`)

// ValidateShellArgSafe reports an error if s contains anything other than a
// plain path/filename character (letters, digits, '.', '/', '-', '_'). This
// is required (not just ValidateDockerfileSafe's narrower quote/newline
// check) for a value embedded into a Dockerfile RUN instruction: RUN runs in
// shell form (`/bin/sh -c "..."`), so shell metacharacters such as ';', '|',
// '&', or '$()' are dangerous there even without any quote or newline
// breakout, since the whole RUN line is itself a shell command.
func ValidateShellArgSafe(s, label string) error {
	if !shellArgSafeRE.MatchString(s) {
		return fmt.Errorf("%s %q contains a character not allowed in a value embedded in a generated Dockerfile RUN instruction; only letters, digits, '.', '/', '-', and '_' are permitted", label, s)
	}
	return nil
}
