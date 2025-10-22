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
	"encoding/json"

	"github.com/a2aproject/a2a-go/a2asrv"
	"google.golang.org/adk/session"
)

type invocationMeta struct {
	userID    string
	sessionID string
	eventMeta map[string]any
}

func toInvocationMeta(config ExecutorConfig, reqCtx *a2asrv.RequestContext) invocationMeta {
	// TODO(yarolegovich): update once A2A provides auth data extraction from Context
	userID, sessionID := "A2A_USER_"+reqCtx.ContextID, reqCtx.ContextID

	m := map[string]any{
		toMetaKey("app_name"):   config.RunnerConfig.AppName,
		toMetaKey("user_id"):    userID,
		toMetaKey("session_id"): sessionID,
	}

	return invocationMeta{userID: userID, sessionID: sessionID, eventMeta: m}
}

func toMetaKey(key string) string {
	return "adk_" + key
}

func toEventMeta(meta invocationMeta, event *session.Event) (map[string]any, error) {
	result := make(map[string]any)
	for k, v := range meta.eventMeta {
		result[k] = v
	}

	for k, v := range map[string]string{
		"invocation_id": event.InvocationID,
		"author":        event.Author,
		"branch":        event.Branch,
	} {
		if v != "" {
			result[toMetaKey(k)] = v
		}
	}

	response := event.LLMResponse

	if response.ErrorCode != "" {
		result[toMetaKey("error_code")] = response.ErrorCode
	}

	if response.GroundingMetadata != nil {
		v, err := toMapStructure(response.GroundingMetadata)
		if err != nil {
			return nil, err
		}
		result[toMetaKey("grounding_metadata")] = v
	}

	// TODO(yarolegovich): include custom and usage metadata when added to session.Event

	return result, nil
}

// We can't use mapstructure in a way compatible with ADK-python, because genai type fields
// don't have proper field tags.
// TODO(yarolegovich): field annotation PR for genai types.
func toMapStructure(data any) (map[string]any, error) {
	bytes, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}

	var result map[string]any
	if err := json.Unmarshal(bytes, &result); err != nil {
		return nil, err
	}
	return result, nil
}
