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

package adka2a

import (
	"google.golang.org/adk/tool"
)

// A2AMetadataProvider creates a function that extracts A2A request metadata
// from the context. The returned function can be used as mcptoolset.MetadataProvider
// to forward A2A metadata to MCP tool calls.
//
// The forwarded metadata includes:
//   - "a2a:task_id": The A2A task ID (if present)
//   - "a2a:context_id": The A2A context ID (if present)
//   - Any keys specified in forwardKeys from request/message metadata
//
// If forwardKeys is nil or empty, all request and message metadata keys are forwarded.
// If forwardKeys is non-empty, only the specified keys are forwarded.
//
// Example usage:
//
//	mcptoolset.New(mcptoolset.Config{
//		Transport: transport,
//		MetadataProvider: adka2a.A2AMetadataProvider(nil), // forward all
//	})
//
//	mcptoolset.New(mcptoolset.Config{
//		Transport: transport,
//		MetadataProvider: adka2a.A2AMetadataProvider([]string{"trace_id"}), // forward specific keys
//	})
func A2AMetadataProvider(forwardKeys []string) func(tool.Context) map[string]any {
	keySet := make(map[string]bool)
	for _, k := range forwardKeys {
		keySet[k] = true
	}

	return func(ctx tool.Context) map[string]any {
		a2aMeta := A2AMetadataFromContext(ctx)
		if a2aMeta == nil {
			return nil
		}

		result := make(map[string]any)

		// Always include task and context IDs if present
		if a2aMeta.TaskID != "" {
			result["a2a:task_id"] = a2aMeta.TaskID
		}
		if a2aMeta.ContextID != "" {
			result["a2a:context_id"] = a2aMeta.ContextID
		}

		// Forward selected or all metadata keys
		forwardMetadata := func(source map[string]any) {
			for k, v := range source {
				if len(keySet) == 0 || keySet[k] {
					result[k] = v
				}
			}
		}

		forwardMetadata(a2aMeta.RequestMetadata)
		forwardMetadata(a2aMeta.MessageMetadata)

		if len(result) == 0 {
			return nil
		}
		return result
	}
}
