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

package llminternal

import (
	"fmt"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/benchutil"
)

// BenchmarkBuildContentsDefault is prompt assembly on its own: the per-turn
// pass over the whole history that every LLM call pays.
func BenchmarkBuildContentsDefault(b *testing.B) {
	for _, n := range []int{10, 100, 1000, 10000} {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			events := benchutil.BenchEvents(n, "agent", time.Unix(1_700_000_000, 0))
			user := genai.NewContentFromText("hello", genai.RoleUser)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := buildContentsDefault("agent", "", "", events, false, user); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
