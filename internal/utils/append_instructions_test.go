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

package utils_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
)

func TestAppendInstructions(t *testing.T) {
	tests := []struct {
		name         string
		instructions []string
		want         *genai.GenerateContentConfig
	}{
		{
			name: "no instructions leave the request untouched",
			want: nil,
		},
		{
			name:         "instructions of one call are joined with a blank line",
			instructions: []string{"First instruction", "Second instruction"},
			want: &genai.GenerateContentConfig{
				SystemInstruction: &genai.Content{
					Role:  genai.RoleUser,
					Parts: []*genai.Part{{Text: "First instruction\n\nSecond instruction"}},
				},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := &model.LLMRequest{}

			utils.AppendInstructions(req, tc.instructions...)

			if diff := cmp.Diff(tc.want, req.Config); diff != "" {
				t.Errorf("req.Config mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
