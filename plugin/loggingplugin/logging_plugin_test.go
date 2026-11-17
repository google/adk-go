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

package loggingplugin

import (
	"testing"

	"google.golang.org/genai"
)

func TestFormatContentTruncatesByCharacter(t *testing.T) {
	p := &loggingPlugin{name: "test"}
	tests := []struct {
		name      string
		text      string
		maxLength int
		want      string
	}{
		{name: "ascii", text: "hello world", maxLength: 5, want: "text: 'hello...'"},
		{name: "multi-byte", text: "日本語のテキスト", maxLength: 4, want: "text: '日本語の...'"},
		{name: "multi-byte fits", text: "日本語", maxLength: 3, want: "text: '日本語'"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := p.formatContent(genai.NewContentFromText(tt.text, genai.RoleUser), tt.maxLength)
			if got != tt.want {
				t.Errorf("formatContent(%q, %d) = %q, want %q", tt.text, tt.maxLength, got, tt.want)
			}
		})
	}
}

func TestFormatArgsTruncatesByCharacter(t *testing.T) {
	p := &loggingPlugin{name: "test"}
	got := p.formatArgs(map[string]any{"q": "日本語"}, 6)
	if want := "map[q:...}"; got != want {
		t.Errorf("formatArgs() = %q, want %q", got, want)
	}
	got = p.formatArgs(map[string]any{"q": "日本語"}, 7)
	if want := "map[q:日...}"; got != want {
		t.Errorf("formatArgs() = %q, want %q", got, want)
	}
}
