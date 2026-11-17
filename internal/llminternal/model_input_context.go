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

package llminternal

import (
	"fmt"
	"reflect"
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/sessionutils"
)

// insertModelInputContext returns contents with the deep copies of
// ctx.RunConfig().ModelInputContext inserted before the last element equal to
// the normalized ctx.UserContent(), else at index 0. It returns contents
// unchanged when ctx.RunConfig() or its ModelInputContext is nil, or the slice
// is empty. It returns the error
// fmt.Errorf("agent.RunConfig.ModelInputContext[%d] is nil", i) for the first
// nil entry i, before it inserts anything. It never edits contents' elements.
//
// The anchor is compared with contents as the history builders and the
// removal of client function call IDs left them, while the user content keeps
// its IDs: adk-python inserts model_input_context after _contents.py copies
// the history without those IDs, and searches for the unstripped user content.
func insertModelInputContext(ctx agent.InvocationContext, contents []*genai.Content) ([]*genai.Content, error) {
	cfg := ctx.RunConfig()
	if cfg == nil || cfg.ModelInputContext == nil || len(*cfg.ModelInputContext) == 0 {
		return contents, nil
	}
	entries := *cfg.ModelInputContext
	copies := make([]*genai.Content, len(entries))
	for i, c := range entries {
		if c == nil {
			return nil, fmt.Errorf("agent.RunConfig.ModelInputContext[%d] is nil", i)
		}
		copies[i] = sessionutils.CloneValue(c)
	}
	return slices.Insert(contents, modelInputContextIndex(ctx.UserContent(), contents), copies...), nil
}

// modelInputContextIndex returns the index of the last element of contents
// equal to userContent without its empty parts, the parts the history
// builders drop. It returns 0 when userContent is nil, has no part left, or
// equals no element.
func modelInputContextIndex(userContent *genai.Content, contents []*genai.Content) int {
	if userContent == nil {
		return 0
	}
	anchor := clone(userContent)
	anchor.Parts = slices.DeleteFunc(anchor.Parts, isEmptyPart)
	if len(anchor.Parts) == 0 {
		return 0
	}
	for i, content := range slices.Backward(contents) {
		if reflect.DeepEqual(content, anchor) {
			return i
		}
	}
	return 0
}
