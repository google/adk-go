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

package geminitool

import (
	"regexp"
	"strings"
)

// Model-id helpers, ported from adk-python v2.0.0
// (google/adk/utils/model_name_utils.py). A bare model name ("gemini-2.5-pro")
// or a publisher/apigee path ("projects/.../models/gemini-2.5-flash") both
// resolve to the model name before matching.
var (
	modelPathPatterns = []*regexp.Regexp{
		regexp.MustCompile(`^projects/[^/]+/locations/[^/]+/publishers/[^/]+/models/(.+)$`),
		regexp.MustCompile(`^apigee/(?:[^/]+/)?(?:[^/]+/)?(.+)$`),
	}
	geminiModelRe  = regexp.MustCompile(`^gemini-`)
	gemini1ModelRe = regexp.MustCompile(`^gemini-1\.\d+`)
)

// extractModelName resolves a simple or path-based model string to its model
// name.
func extractModelName(modelString string) string {
	for _, p := range modelPathPatterns {
		if m := p.FindStringSubmatch(modelString); m != nil {
			return m[1]
		}
	}
	return strings.TrimPrefix(modelString, "models/")
}

// isGeminiModel reports whether the model string names a Gemini model.
func isGeminiModel(modelString string) bool {
	if modelString == "" {
		return false
	}
	return geminiModelRe.MatchString(extractModelName(modelString))
}

// isGemini1Model reports whether the model string names a Gemini 1.x model.
func isGemini1Model(modelString string) bool {
	if modelString == "" {
		return false
	}
	return gemini1ModelRe.MatchString(extractModelName(modelString))
}
