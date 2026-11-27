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

package sqlitememory

import (
	"testing"

	"google.golang.org/genai"
)

func TestEventSearchText(t *testing.T) {
	for _, tc := range []struct {
		name     string
		parts    []*genai.Part
		maxBytes int
		want     string
	}{
		{
			name:     "visible text only",
			parts:    []*genai.Part{nil, {Text: "ignore", Thought: true}, {Text: " Tea\nTime "}, {Text: " Notes "}},
			maxBytes: 40,
			want:     "tea time notes",
		},
		{
			name:     "normalize before splitting and lowercasing",
			parts:    []*genai.Part{{Text: "\x1cCafe\u0301乌龙İSTANBUL"}},
			maxBytes: 40,
			want:     "café 乌龙 i\u0307stanbul",
		},
		{
			name:     "truncate at UTF-8 boundary",
			parts:    []*genai.Part{{Text: "乌龙茶"}},
			maxBytes: 7,
			want:     "乌龙",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got, visible := eventSearchText(&genai.Content{Parts: tc.parts}, tc.maxBytes); !visible || got != tc.want {
				t.Error("unexpected normalized search text")
			}
		})
	}
}
