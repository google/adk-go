// Copyright 2025 The A2A Authors
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

package jsonrpc

import (
	"errors"
	"fmt"

	"github.com/a2aproject/a2a-go/a2a"
)

// JSON-RPC 2.0 protocol constants
const (
	Version = "2.0"

	// HTTP headers
	ContentJSON = "application/json"

	// JSON-RPC method names per A2A spec §7
	MethodMessageSend          = "message/send"
	MethodMessageStream        = "message/stream"
	MethodTasksGet             = "tasks/get"
	MethodTasksCancel          = "tasks/cancel"
	MethodTasksResubscribe     = "tasks/resubscribe"
	MethodPushConfigGet        = "tasks/pushNotificationConfig/get"
	MethodPushConfigSet        = "tasks/pushNotificationConfig/set"
	MethodPushConfigList       = "tasks/pushNotificationConfig/list"
	MethodPushConfigDelete     = "tasks/pushNotificationConfig/delete"
	MethodGetExtendedAgentCard = "agent/getAuthenticatedExtendedCard"
)

// jsonrpcError represents a JSON-RPC 2.0 error object.
// TODO(yarolegovich): Convert to transport-agnostic error format so Client can use errors.Is(err, a2a.ErrMethodNotFound).
// This needs to be implemented across all transports (currently not in grpc either).
type Error struct {
	Code    int            `json:"code"`
	Message string         `json:"message"`
	Data    map[string]any `json:"data,omitempty"`
}

// Error implements the error interface for jsonrpcError.
func (e *Error) Error() string {
	if len(e.Data) > 0 {
		return fmt.Sprintf("jsonrpc error %d: %s (data: %v)", e.Code, e.Message, e.Data)
	}
	return fmt.Sprintf("jsonrpc error %d: %s", e.Code, e.Message)
}

var codeToError = map[int]error{
	-32700: a2a.ErrParseError,
	-32600: a2a.ErrInvalidRequest,
	-32601: a2a.ErrMethodNotFound,
	-32602: a2a.ErrInvalidParams,
	-32603: a2a.ErrInternalError,
	-32000: a2a.ErrServerError,
	-32001: a2a.ErrTaskNotFound,
	-32002: a2a.ErrTaskNotCancelable,
	-32003: a2a.ErrPushNotificationNotSupported,
	-32004: a2a.ErrUnsupportedOperation,
	-32005: a2a.ErrUnsupportedContentType,
	-32006: a2a.ErrInvalidAgentResponse,
	-32007: a2a.ErrAuthenticatedExtendedCardNotConfigured,
}

func (e *Error) ToA2AError() error {
	err, ok := codeToError[e.Code]
	if !ok {
		err = a2a.ErrInternalError
	}
	if e.Data == nil {
		return err
	}
	extra, ok := e.Data["error"].(string)
	if !ok {
		return err
	}
	return fmt.Errorf("%s: %w", extra, err)

}

func ToJSONRPCError(err error) *Error {
	jsonrpcErr := &Error{}
	if errors.As(err, &jsonrpcErr) {
		return jsonrpcErr
	}

	for code, a2aErr := range codeToError {
		if errors.Is(err, a2aErr) {
			return &Error{
				Code:    code,
				Message: a2aErr.Error(),
				Data:    map[string]any{"error": err.Error()},
			}
		}
	}
	return &Error{Code: -32603, Message: a2a.ErrInternalError.Error()}
}
