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

package googlellm

import (
	"os"
	"regexp"
	"strings"
)

// disableGeminiModelIDCheckEnvVar turns the Gemini model name check of the
// built-in code executor off, as in adk-python utils/model_name_utils.py.
const disableGeminiModelIDCheckEnvVar = "ADK_DISABLE_GEMINI_MODEL_ID_CHECK"

// modelPathPatterns are the path forms of adk-python's extract_model_name, in
// its order. Python's "$" also matches before one trailing newline, which Go's
// does not, so each pattern allows that newline explicitly.
var modelPathPatterns = []*regexp.Regexp{
	regexp.MustCompile(`^projects/[^/]+/locations/[^/]+/publishers/[^/]+/models/(.+)\n?$`),
	regexp.MustCompile(`^apigee/(?:[^/]+/)?(?:[^/]+/)?(.+)\n?$`),
}

// IsGeminiModelExact reports whether model names a Gemini model by Python's
// rule (adk-python utils/model_name_utils.py is_gemini_model): the name
// extracted from model starts with "gemini-". Unlike IsGeminiModel it is
// case-sensitive and does not reduce a malformed "projects/" path, so
// "Gemini-2.5-flash", "projects/123/locations/us-central1/models/gemini-2.5-flash"
// and "apigee/a/b/c/gemini-2.5-flash" are not Gemini names, while "gemini-1.5-pro",
// "models/gemini-2.5-pro", a full Vertex path, "apigee/v1/gemini-2.5-flash",
// "gemini/gemini-2.5-flash" and "openrouter/google/gemini-2.5-pro:online" are.
// The empty string is not. It has no side effects and is safe for concurrent use.
func IsGeminiModelExact(model string) bool {
	return strings.HasPrefix(extractModelNameExact(model), "gemini-")
}

// extractModelNameExact returns the model name in model step by step as
// adk-python's extract_model_name does: the first step that applies decides
// the result.
func extractModelNameExact(model string) string {
	for _, p := range modelPathPatterns {
		if m := p.FindStringSubmatch(model); m != nil {
			return m[1]
		}
	}
	if name, ok := strings.CutPrefix(model, "models/"); ok {
		return name
	}
	// A "projects/" path that is not a full Vertex AI path stays as it is, so
	// the provider-prefix step cannot read its last segment as a Gemini name.
	if strings.HasPrefix(model, "projects/") {
		return model
	}
	// A provider prefix ("gemini/gemini-2.5-flash") is dropped only when the
	// last segment is a Gemini name.
	if i := strings.LastIndex(model, "/"); i >= 0 && strings.HasPrefix(model[i+1:], "gemini-") {
		return model[i+1:]
	}
	return model
}

// IsGeminiModelIDCheckDisabled reports whether the environment variable
// ADK_DISABLE_GEMINI_MODEL_ID_CHECK, read on this call and not cached, holds
// "true" or "1" in any letter case. Any other value, an empty value and an
// unset variable return false. It logs nothing and is safe for concurrent use.
func IsGeminiModelIDCheckDisabled() bool {
	v := strings.ToLower(os.Getenv(disableGeminiModelIDCheckEnvVar))
	return v == "true" || v == "1"
}
