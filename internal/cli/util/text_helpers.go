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
	"strings"
	"unicode/utf8"
)

func CenterString(s string, w int) string {
	sw := w - len(s)
	lw := sw / 2
	rw := sw - lw
	return strings.Repeat(" ", lw) + s + strings.Repeat(" ", rw)
}

// ValidateDockerfileSafe reports an error if s contains a byte that, embedded
// literally into generated Dockerfile content, could break out of the
// double-quoted string it sits in or out of the current instruction entirely.
// Dockerfile instructions are newline-delimited and offer no generic escaping
// mechanism for interpolated values, so the check rejects rather than quotes.
//
// The rejected set is '"', '`', '\', every C0 control byte (which covers NUL,
// LF and CR), and any string that is not valid UTF-8. Backslash and the
// remaining control bytes are part of it because an exec-form CMD must be a
// valid JSON array, and Docker's response to an unparseable exec form is not to
// fail the build but to silently reinterpret the whole instruction as shell
// form. A value that does parse can still be wrong: JSON decodes, say, "\n" into
// a real newline, and an invalid UTF-8 byte is replaced with U+FFFD by both the
// Dockerfile parser and Go's JSON decoder, so the container would receive
// something other than what was written. Accepting a value therefore guarantees
// the emitted CMD line still parses as JSON, and that the value survives into
// the container unchanged.
//
// This is the weaker of the two checks. A value that reaches a RUN instruction
// needs ValidateShellArgSafe instead.
func ValidateDockerfileSafe(s, label string) error {
	// Deliberately a byte loop rather than a rune loop: the property being
	// enforced is about the bytes written to the file.
	for i := 0; i < len(s); i++ {
		if c := s[i]; c == '"' || c == '`' || c == '\\' || c < 0x20 {
			return fmt.Errorf("%s %q contains a character (quote, backtick, backslash, or a control character) that is not allowed in a value embedded in generated Dockerfile content", label, s)
		}
	}
	if !utf8.ValidString(s) {
		return fmt.Errorf("%s %q is not valid UTF-8; the invalid bytes would reach the container as U+FFFD rather than as written", label, s)
	}
	return nil
}

// ValidateShellArgSafe reports an error if s contains anything other than a
// plain path/filename character: letters, digits, '.', '/', '-', '_', or a
// non-ASCII rune. It is the check for a value that reaches a Dockerfile RUN
// instruction, which runs in shell form (`/bin/sh -c "..."`), so shell
// metacharacters such as ';', '|', '&', or '$()' are dangerous there even
// without any quote or newline breakout. It is also the check for a value that
// lands in a COPY operand, since COPY splits its operands on whitespace, which
// ValidateDockerfileSafe permits.
//
// Non-ASCII characters are admitted because excluding them buys no safety and
// does break working deploys. Every byte of a multi-byte UTF-8 rune is >= 0x80
// while every shell metacharacter, and every byte the Dockerfile parser treats
// as a token separator, is ASCII. Meanwhile the values checked here include a
// whole --entry_point_path, whose directory components routinely carry
// characters no Go filename would. ASCII punctuation outside the class stays
// rejected even where it is inert today, so the allowlist does not depend on
// where a future caller interpolates the value.
func ValidateShellArgSafe(s, label string) error {
	// A byte loop, for the same reason as in ValidateDockerfileSafe. Every byte
	// of a multi-byte rune is >= 0x80, so the class can be applied per byte and
	// the UTF-8 check below rules out a stray high byte that is not part of one.
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9':
		case c == '_', c == '.', c == '/', c == '-':
		case c >= 0x80:
		default:
			return fmt.Errorf("%s %q contains a character not allowed in a value embedded in generated Dockerfile content; whitespace splits COPY operands and shell metacharacters are live in a RUN line, so only letters, digits, '.', '/', '-', '_', and non-ASCII characters are permitted", label, s)
		}
	}
	if !utf8.ValidString(s) {
		return fmt.Errorf("%s %q is not valid UTF-8; a COPY operand containing an invalid byte does not match the file it names", label, s)
	}
	return nil
}
