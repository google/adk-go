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

func toInvocationMeta(config *ExecutorConfig, reqCtx a2asrv.RequestContext) invocationMeta {
	// TODO(yarolegovich): update once A2A provides auth data extraction from Context
	userID, sessionID := "A2A_USER_"+reqCtx.ContextID, reqCtx.ContextID

	m := map[string]any{
		toMetaKey("app_name"):   config.AppName,
		toMetaKey("user_id"):    userID,
		toMetaKey("session_id"): sessionID,
	}

	return invocationMeta{userID: userID, sessionID: sessionID, eventMeta: m}
}

func toMetaKey(key string) string {
	return "adk_" + key
}

func toEventMeta(meta invocationMeta, event *session.Event) (map[string]any, error) {
	result := make(map[string]any, len(meta.eventMeta)+5)

	for k, v := range meta.eventMeta {
		result[k] = v
	}

	result[toMetaKey("invocation_id")] = event.InvocationID
	result[toMetaKey("author")] = event.Author

	if event.Branch != "" {
		result[toMetaKey("branch")] = event.Branch
	}

	response := event.LLMResponse
	if response == nil {
		return result, nil
	}

	if response.ErrorCode != 0 {
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
	if err := json.Unmarshal(bytes, result); err != nil {
		return nil, err
	}
	return result, nil
}
