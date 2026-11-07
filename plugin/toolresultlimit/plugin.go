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

// Package toolresultlimit provides a plugin that limits the JSON size of tool results.
package toolresultlimit

import (
	"encoding/json"
	"fmt"
	"unicode/utf8"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/tool"
)

const minResultBytes = 256

// Config configures the tool result limit.
type Config struct {
	// MaxResultBytes is the JSON byte limit, including metadata. Minimum: 256.
	MaxResultBytes int
}

// New creates an AfterToolCallback plugin that replaces oversized results with
// marked JSON previews. Tool errors are left unchanged. Callback short-circuit
// rules apply; streaming tools are not covered.
func New(cfg Config) (*plugin.Plugin, error) {
	if cfg.MaxResultBytes < minResultBytes {
		return nil, fmt.Errorf("MaxResultBytes must be at least %d", minResultBytes)
	}
	return plugin.New(plugin.Config{
		Name: "ToolResultLimitPlugin",
		AfterToolCallback: func(_ agent.Context, _ tool.Tool, _, result map[string]any, toolErr error) (map[string]any, error) {
			// Returning a result with nil error would clear toolErr.
			if toolErr != nil {
				return nil, nil
			}
			return limitResult(result, cfg.MaxResultBytes)
		},
	})
}

// limitResult returns nil when no replacement is needed.
func limitResult(result map[string]any, maxBytes int) (map[string]any, error) {
	encoded, err := json.Marshal(result)
	if err != nil {
		return nil, &encodingError{cause: err}
	}
	if len(encoded) <= maxBytes {
		return nil, nil
	}

	output := map[string]any{
		"truncated":           true,
		"original_size_bytes": len(encoded),
		"message":             "Partial result. Narrow the query or request a smaller page.",
		"preview":             "",
	}
	response := map[string]any{"output": output}
	if result["error"] != nil {
		response["error"] = "Tool reported an error; details are truncated in output.preview."
	}

	// Keep the search bounds unchanged when rounding to a UTF-8 boundary.
	best := ""
	for low, high := 0, maxBytes; low <= high; {
		mid := low + (high-low)/2
		end := mid
		for end > 0 && !utf8.RuneStart(encoded[end]) {
			end--
		}
		preview := string(encoded[:end])
		output["preview"] = preview
		candidate, _ := json.Marshal(response) // All values are JSON primitives.
		if len(candidate) <= maxBytes {
			best = preview
			low = mid + 1
		} else {
			high = mid - 1
		}
	}
	output["preview"] = best
	return response, nil
}

// Preserve the cause without exposing result data in the error message.
type encodingError struct {
	cause error
}

func (*encodingError) Error() string   { return "cannot JSON-encode tool result for size limit" }
func (e *encodingError) Unwrap() error { return e.cause }
