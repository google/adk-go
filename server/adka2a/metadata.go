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
	"context"
	"maps"

	"github.com/a2aproject/a2a-go/a2asrv"

	"google.golang.org/adk/internal/converters"
	"google.golang.org/adk/session"
)

// ToA2AMetaKey adds a prefix used to differentiage ADK-related values stored in Metadata an A2A event.
func ToA2AMetaKey(key string) string {
	return "adk_" + key
}

type invocationMeta struct {
	userID    string
	sessionID string
	eventMeta map[string]any
}

func toInvocationMeta(ctx context.Context, config ExecutorConfig, reqCtx *a2asrv.RequestContext) invocationMeta {
	// TODO(yarolegovich): update once A2A provides auth data extraction from Context
	userID, sessionID := "A2A_USER_"+reqCtx.ContextID, reqCtx.ContextID

	// override userID if set in the call context
	if callCtx, ok := a2asrv.CallContextFrom(ctx); ok {
		if callCtx.User != nil && callCtx.User.Name() != "" {
			userID = callCtx.User.Name()
		}
	}

	m := map[string]any{
		ToA2AMetaKey("app_name"):   config.RunnerConfig.AppName,
		ToA2AMetaKey("user_id"):    userID,
		ToA2AMetaKey("session_id"): sessionID,
	}

	return invocationMeta{userID: userID, sessionID: sessionID, eventMeta: m}
}

func toEventMeta(meta invocationMeta, event *session.Event) (map[string]any, error) {
	result := make(map[string]any)
	maps.Copy(result, meta.eventMeta)

	for k, v := range map[string]string{
		"invocation_id": event.InvocationID,
		"author":        event.Author,
		"branch":        event.Branch,
	} {
		if v != "" {
			result[ToA2AMetaKey(k)] = v
		}
	}

	response := event.LLMResponse

	if response.ErrorCode != "" {
		result[ToA2AMetaKey("error_code")] = response.ErrorCode
	}

	if response.GroundingMetadata != nil {
		v, err := converters.ToMapStructure(response.GroundingMetadata)
		if err != nil {
			return nil, err
		}
		result[ToA2AMetaKey("grounding_metadata")] = v
	}

	// TODO(yarolegovich): include custom and usage metadata when added to session.Event

	return result, nil
}
