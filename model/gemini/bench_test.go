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

package gemini_test

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/benchutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/tool/toolutils"
)

// BenchmarkGenerateContent is the client-side cost of one Gemini call whose
// prompt carries n history contents and three tool declarations, the
// request a tool turn sends twice.
func BenchmarkGenerateContent(b *testing.B) {
	m, err := gemini.NewModel(b.Context(), "gemini-2.5-flash", &genai.ClientConfig{
		APIKey:     "bench-key",
		Backend:    genai.BackendGeminiAPI,
		HTTPClient: &http.Client{Transport: &benchutil.GeminiTransport{Bodies: [][]byte{benchutil.GeminiText}}},
	})
	if err != nil {
		b.Fatal(err)
	}
	tools, err := benchutil.Tools()
	if err != nil {
		b.Fatal(err)
	}
	for _, n := range []int{0, 100, 1000} {
		b.Run(fmt.Sprintf("contents=%d", n), func(b *testing.B) {
			req := &model.LLMRequest{Model: "gemini-2.5-flash", Config: &genai.GenerateContentConfig{
				SystemInstruction: genai.NewContentFromText(benchutil.ToolInstruction, genai.RoleUser),
			}}
			for _, t := range tools {
				if err := toolutils.PackTool(req, t.(toolutils.Tool)); err != nil {
					b.Fatal(err)
				}
			}
			for _, ev := range benchutil.BenchEvents(n, "agent", time.Unix(1_700_000_000, 0)) {
				req.Contents = append(req.Contents, ev.Content)
			}
			// Ending on a user turn keeps GenerateContent from appending a
			// continuation prompt to req on every iteration.
			req.Contents = append(req.Contents, genai.NewContentFromText("What is the weather in Paris?", genai.RoleUser))
			b.ReportAllocs()
			for b.Loop() {
				for _, err := range m.GenerateContent(b.Context(), req, false) {
					if err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}
