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

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"

	"google.golang.org/adk/internal/converters"
	"google.golang.org/adk/session"
)

var (
	customMetaTaskIDKey    = ToADKMetaKey("task_id")
	customMetaContextIDKey = ToADKMetaKey("context_id")

	metadataEscalateKey        = ToA2AMetaKey("escalate")
	metadataTransferToAgentKey = ToA2AMetaKey("transfer_to_agent")
	metadataErrorCodeKey       = ToA2AMetaKey("error_code")
	metadataCitationKey        = ToA2AMetaKey("citation_metadata")
	metadataGroundingKey       = ToA2AMetaKey("grounding_metadata")
	metadataUsageKey           = ToA2AMetaKey("usage_metadata")
	metadataCustomMetaKey      = ToA2AMetaKey("custom_metadata")
	metadataPartialKey         = ToA2AMetaKey("partial")
)

// ToA2AMetaKey adds a prefix used to differentiage ADK-related values stored in Metadata an A2A event.
func ToA2AMetaKey(key string) string {
	return "adk_" + key
}

// ToADKMetaKey adds a prefix used to differentiage A2A-related values stored in custom metadata of an ADK session event.
func ToADKMetaKey(key string) string {
	return "a2a:" + key
}

type invocationMeta struct {
	userID    string
	sessionID string
	agentName string
	reqCtx    *a2asrv.RequestContext
	eventMeta map[string]any
}

func toInvocationMeta(ctx context.Context, config ExecutorConfig, reqCtx *a2asrv.RequestContext) invocationMeta {
	userID, sessionID := "A2A_USER_"+reqCtx.ContextID, reqCtx.ContextID

	// a2a sdk attaches authn info to the call context, use it when provided
	if callCtx, ok := a2asrv.CallContextFrom(ctx); ok {
		if callCtx.User != nil && callCtx.User.Name() != "" {
			userID = callCtx.User.Name()
		}
	}

	meta := map[string]any{
		ToA2AMetaKey("app_name"):   config.RunnerConfig.AppName,
		ToA2AMetaKey("user_id"):    userID,
		ToA2AMetaKey("session_id"): sessionID,
	}

	return invocationMeta{
		userID:    userID,
		sessionID: sessionID,
		agentName: config.RunnerConfig.Agent.Name(),
		eventMeta: meta,
		reqCtx:    reqCtx,
	}
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

	if err := addMeta(result, metadataCitationKey, event.CitationMetadata); err != nil {
		return nil, err
	}
	if err := addMeta(result, metadataGroundingKey, event.GroundingMetadata); err != nil {
		return nil, err
	}
	if err := addMeta(result, metadataUsageKey, event.UsageMetadata); err != nil {
		return nil, err
	}
	if event.CustomMetadata != nil {
		result[metadataCustomMetaKey] = event.CustomMetadata
	}
	if event.LLMResponse.ErrorCode != "" {
		result[metadataErrorCodeKey] = event.LLMResponse.ErrorCode
	}

	return result, nil
}

func setActionsMeta(meta map[string]any, actions session.EventActions) map[string]any {
	if actions.TransferToAgent == "" && !actions.Escalate { // if meta was nil, it should remain nil
		return meta
	}
	if meta == nil {
		meta = map[string]any{}
	}
	if actions.Escalate {
		meta[metadataEscalateKey] = true
	}
	if actions.TransferToAgent != "" {
		meta[metadataTransferToAgentKey] = actions.TransferToAgent
	}
	return meta
}

func processA2AMeta(a2aEvent a2a.Event, event *session.Event) error {
	taskInfo, meta := a2aEvent.TaskInfo(), a2aEvent.Meta()

	if err := processMeta(metadataCitationKey, meta, &event.CitationMetadata); err != nil {
		return err
	}
	if err := processMeta(metadataGroundingKey, meta, &event.GroundingMetadata); err != nil {
		return err
	}
	if err := processMeta(metadataUsageKey, meta, &event.UsageMetadata); err != nil {
		return err
	}

	event.CustomMetadata = ToCustomMetadata(taskInfo.TaskID, taskInfo.ContextID)
	if um, ok := meta[metadataCustomMetaKey].(map[string]any); ok {
		if event.CustomMetadata == nil {
			event.CustomMetadata = make(map[string]any)
		}
		maps.Copy(event.CustomMetadata, um)
	}

	if ec, ok := meta[metadataErrorCodeKey].(string); ok {
		event.LLMResponse.ErrorCode = ec
	}

	event.Actions = toEventActions(a2aEvent.Meta())
	return nil
}

func addMeta(result map[string]any, key string, data any) error {
	if data == nil {
		return nil
	}
	v, err := converters.ToMapStructure(data)
	if err != nil {
		return err
	}
	if v == nil {
		return nil
	}

	result[key] = v
	return nil
}

func processMeta[T any](key string, meta map[string]any, target **T) error {
	if m, ok := meta[key].(map[string]any); ok {
		converted, err := converters.FromMapStructure[T](m)
		if err != nil {
			return err
		}
		*target = converted
	}
	return nil
}
