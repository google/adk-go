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

package mcptoolset

import (
	"google.golang.org/adk/tool"
)

// SessionStateMetadataProvider creates a MetadataProvider that reads metadata
// from session state keys. This is useful for non-A2A scenarios where metadata
// is stored in session state.
//
// The stateKeys map specifies which session state keys to read and how to name
// them in the MCP metadata. For example:
//
//	mcptoolset.New(mcptoolset.Config{
//		Transport: transport,
//		MetadataProvider: mcptoolset.SessionStateMetadataProvider(map[string]string{
//			"temp:trace_id":   "x-trace-id",
//			"temp:request_id": "x-request-id",
//		}),
//	})
//
// This would read "temp:trace_id" from state and forward it as "x-trace-id" in MCP metadata.
func SessionStateMetadataProvider(stateKeys map[string]string) MetadataProvider {
	return func(ctx tool.Context) map[string]any {
		if len(stateKeys) == 0 {
			return nil
		}

		result := make(map[string]any)
		state := ctx.ReadonlyState()

		for stateKey, metaKey := range stateKeys {
			if val, err := state.Get(stateKey); err == nil {
				result[metaKey] = val
			}
		}

		if len(result) == 0 {
			return nil
		}
		return result
	}
}

// ChainMetadataProviders combines multiple MetadataProviders into one.
// Each provider is called in order, and later providers can override
// keys set by earlier providers.
//
// Example usage:
//
//	mcptoolset.New(mcptoolset.Config{
//		Transport: transport,
//		MetadataProvider: mcptoolset.ChainMetadataProviders(
//			adka2a.A2AMetadataProvider(nil),
//			mcptoolset.SessionStateMetadataProvider(map[string]string{
//				"temp:custom_field": "custom-field",
//			}),
//		),
//	})
func ChainMetadataProviders(providers ...MetadataProvider) MetadataProvider {
	return func(ctx tool.Context) map[string]any {
		result := make(map[string]any)
		for _, p := range providers {
			if p == nil {
				continue
			}
			if meta := p(ctx); meta != nil {
				for k, v := range meta {
					result[k] = v
				}
			}
		}
		if len(result) == 0 {
			return nil
		}
		return result
	}
}
