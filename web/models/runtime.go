package models

import (
	"fmt"
	"net/http"

	"google.golang.org/adk/web/handlers"
	"google.golang.org/adk/web/helpers"
	"google.golang.org/genai"
)

type RunAgentRequest struct {
	AppName string `json:"appName"`

	UserId string `json:"userId"`

	SessionId string `json:"sessionId"`

	NewMessage genai.Content `json:"newMessage"`

	Streaming bool `json:"streaming,omitempty"`

	StateDelta *map[string]interface{} `json:"stateDelta,omitempty"`
}

// AssertRunAgentRequestRequired checks if the required fields are not zero-ed
func AssertRunAgentRequestRequired(obj RunAgentRequest) error {
	elements := map[string]interface{}{
		"appName":    obj.AppName,
		"userId":     obj.UserId,
		"sessionId":  obj.SessionId,
		"newMessage": obj.NewMessage,
	}
	for name, el := range elements {
		if isZero := helpers.IsZeroValue(el); isZero {
			return handlers.StatusError{error: fmt.Errorf("%s is required", name), Code: http.StatusBadRequest}
		}
	}

	return nil
}
