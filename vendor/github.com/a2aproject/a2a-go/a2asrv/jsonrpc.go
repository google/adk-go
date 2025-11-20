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

package a2asrv

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/internal/jsonrpc"
	"github.com/a2aproject/a2a-go/internal/sse"
	"github.com/a2aproject/a2a-go/log"
)

// jsonrpcRequest represents a JSON-RPC 2.0 request.
type jsonrpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
	// ID can be a number, a string or nil.
	ID any `json:"id"`
}

// jsonrpcResponse represents a JSON-RPC 2.0 response.
type jsonrpcResponse struct {
	JSONRPC string         `json:"jsonrpc"`
	ID      any            `json:"id"`
	Result  any            `json:"result,omitempty"`
	Error   *jsonrpc.Error `json:"error,omitempty"`
}

type JSONRPCHandler struct {
	handler RequestHandler
}

func NewJSONRPCHandler(handler RequestHandler) *JSONRPCHandler {
	return &JSONRPCHandler{handler: handler}
}

func (h *JSONRPCHandler) ServeHTTP(rw http.ResponseWriter, req *http.Request) {
	ctx := req.Context()

	if req.Method != "POST" {
		h.writeJSONRPCError(ctx, rw, a2a.ErrInvalidRequest, nil)
		return
	}

	defer func() {
		if err := req.Body.Close(); err != nil {
			log.Error(ctx, "failed to close request body", err)
		}
	}()

	var payload jsonrpcRequest
	if err := json.NewDecoder(req.Body).Decode(&payload); err != nil {
		h.writeJSONRPCError(ctx, rw, newParseError(err), nil)
		return
	}

	if payload.JSONRPC != jsonrpc.Version {
		h.writeJSONRPCError(ctx, rw, a2a.ErrInvalidRequest, nil)
		return
	}

	if payload.Method == jsonrpc.MethodTasksResubscribe || payload.Method == jsonrpc.MethodMessageStream {
		h.handleStreamingRequest(ctx, rw, &payload)
	} else {
		h.handleRequest(ctx, rw, &payload)
	}
}

func (h *JSONRPCHandler) handleRequest(ctx context.Context, rw http.ResponseWriter, req *jsonrpcRequest) {
	var result any
	var err error
	switch req.Method {
	case jsonrpc.MethodTasksGet:
		result, err = h.onGetTask(ctx, req.Params)
	case jsonrpc.MethodMessageSend:
		result, err = h.onSendMessage(ctx, req.Params)
	case jsonrpc.MethodTasksCancel:
		result, err = h.onCancelTask(ctx, req.Params)
	case jsonrpc.MethodPushConfigGet:
		result, err = h.onGetTaskPushConfig(ctx, req.Params)
	case jsonrpc.MethodPushConfigList:
		result, err = h.onListTaskPushConfig(ctx, req.Params)
	case jsonrpc.MethodPushConfigSet:
		result, err = h.onSetTaskPushConfig(ctx, req.Params)
	case jsonrpc.MethodPushConfigDelete:
		err = h.onDeleteTaskPushConfig(ctx, req.Params)
	case jsonrpc.MethodGetExtendedAgentCard:
		result, err = h.onGetAgentCard(ctx)
	default:
		err = a2a.ErrMethodNotFound
	}

	if err != nil {
		h.writeJSONRPCError(ctx, rw, err, req.ID)
		return
	}

	if result != nil {
		resp := jsonrpcResponse{JSONRPC: jsonrpc.Version, ID: req.ID, Result: result}
		if err := json.NewEncoder(rw).Encode(resp); err != nil {
			log.Error(ctx, "failed to encode response", err)
		}
	}
}

func (h *JSONRPCHandler) handleStreamingRequest(ctx context.Context, rw http.ResponseWriter, req *jsonrpcRequest) {
	sseWriter, err := sse.NewWriter(rw)
	if err != nil {
		h.writeJSONRPCError(ctx, rw, err, req.ID)
		return
	}

	sseWriter.WriteHeaders()

	sseChan := make(chan []byte)
	requestCtx, cancelReqCtx := context.WithCancel(ctx)
	defer cancelReqCtx()
	go func() {
		var events iter.Seq2[a2a.Event, error]
		switch req.Method {
		case jsonrpc.MethodTasksResubscribe:
			events = h.onResubscribeToTask(requestCtx, req.Params)
		case jsonrpc.MethodMessageStream:
			events = h.onSendMessageStream(requestCtx, req.Params)
		default:
			events = func(yield func(a2a.Event, error) bool) { yield(nil, a2a.ErrMethodNotFound) }
		}
		eventSeqToSSEDataStream(requestCtx, req, sseChan, events)
	}()

	keepAliveTicker := time.NewTicker(30 * time.Second)
	defer keepAliveTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-keepAliveTicker.C:
			if err := sseWriter.WriteKeepAlive(ctx); err != nil {
				log.Error(ctx, "failed to send keep-alive", err)
				return
			}
		case data, ok := <-sseChan:
			if !ok {
				return
			}
			if err := sseWriter.WriteData(ctx, data); err != nil {
				log.Error(ctx, "failed to write an event", err)
				return
			}
		}
	}

}

func eventSeqToSSEDataStream(ctx context.Context, req *jsonrpcRequest, sseChan chan []byte, events iter.Seq2[a2a.Event, error]) {
	defer close(sseChan)

	handleError := func(err error) {
		jsonrpcErr := jsonrpc.ToJSONRPCError(err)
		resp := jsonrpcResponse{JSONRPC: jsonrpc.Version, ID: req.ID, Error: jsonrpcErr}
		bytes, err := json.Marshal(resp)
		if err != nil {
			log.Error(ctx, "failed to marshal error response", err)
			return
		}
		select {
		case <-ctx.Done():
			return
		case sseChan <- bytes:
		}
	}

	for event, err := range events {
		if err != nil {
			handleError(err)
			return
		}

		resp := jsonrpcResponse{JSONRPC: jsonrpc.Version, ID: req.ID, Result: event}
		bytes, err := json.Marshal(resp)
		if err != nil {
			handleError(err)
			return
		}

		select {
		case <-ctx.Done():
			return
		case sseChan <- bytes:
		}
	}
}

func (h *JSONRPCHandler) onGetTask(ctx context.Context, raw json.RawMessage) (*a2a.Task, error) {
	var query a2a.TaskQueryParams
	if err := json.Unmarshal(raw, &query); err != nil {
		return nil, newParseError(err)
	}
	return h.handler.OnGetTask(ctx, &query)
}

func (h *JSONRPCHandler) onCancelTask(ctx context.Context, raw json.RawMessage) (*a2a.Task, error) {
	var id a2a.TaskIDParams
	if err := json.Unmarshal(raw, &id); err != nil {
		return nil, newParseError(err)
	}
	return h.handler.OnCancelTask(ctx, &id)
}

func (h *JSONRPCHandler) onSendMessage(ctx context.Context, raw json.RawMessage) (a2a.SendMessageResult, error) {
	var message a2a.MessageSendParams
	if err := json.Unmarshal(raw, &message); err != nil {
		return nil, newParseError(err)
	}
	return h.handler.OnSendMessage(ctx, &message)
}

func (h *JSONRPCHandler) onResubscribeToTask(ctx context.Context, raw json.RawMessage) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		var id a2a.TaskIDParams
		if err := json.Unmarshal(raw, &id); err != nil {
			yield(nil, newParseError(err))
			return
		}
		for event, err := range h.handler.OnResubscribeToTask(ctx, &id) {
			if !yield(event, err) {
				return
			}
		}
	}
}

func (h *JSONRPCHandler) onSendMessageStream(ctx context.Context, raw json.RawMessage) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		var message a2a.MessageSendParams
		if err := json.Unmarshal(raw, &message); err != nil {
			yield(nil, newParseError(err))
			return
		}
		for event, err := range h.handler.OnSendMessageStream(ctx, &message) {
			if !yield(event, err) {
				return
			}
		}
	}

}

func (h *JSONRPCHandler) onGetTaskPushConfig(ctx context.Context, raw json.RawMessage) (*a2a.TaskPushConfig, error) {
	var params a2a.GetTaskPushConfigParams
	if err := json.Unmarshal(raw, &params); err != nil {
		return nil, newParseError(err)
	}
	return h.handler.OnGetTaskPushConfig(ctx, &params)
}

func (h *JSONRPCHandler) onListTaskPushConfig(ctx context.Context, raw json.RawMessage) ([]*a2a.TaskPushConfig, error) {
	var params a2a.ListTaskPushConfigParams
	if err := json.Unmarshal(raw, &params); err != nil {
		return nil, newParseError(err)
	}
	return h.handler.OnListTaskPushConfig(ctx, &params)
}

func (h *JSONRPCHandler) onSetTaskPushConfig(ctx context.Context, raw json.RawMessage) (*a2a.TaskPushConfig, error) {
	var params a2a.TaskPushConfig
	if err := json.Unmarshal(raw, &params); err != nil {
		return nil, newParseError(err)
	}
	return h.handler.OnSetTaskPushConfig(ctx, &params)
}

func (h *JSONRPCHandler) onDeleteTaskPushConfig(ctx context.Context, raw json.RawMessage) error {
	var params a2a.DeleteTaskPushConfigParams
	if err := json.Unmarshal(raw, &params); err != nil {
		return newParseError(err)
	}
	return h.handler.OnDeleteTaskPushConfig(ctx, &params)
}

func (h *JSONRPCHandler) onGetAgentCard(ctx context.Context) (*a2a.AgentCard, error) {
	return h.handler.OnGetExtendedAgentCard(ctx)
}

func newParseError(cause error) error {
	return fmt.Errorf("%w: %w", a2a.ErrParseError, cause)
}

func (h *JSONRPCHandler) writeJSONRPCError(ctx context.Context, rw http.ResponseWriter, err error, reqID any) {
	jsonrpcErr := jsonrpc.ToJSONRPCError(err)
	resp := jsonrpcResponse{JSONRPC: jsonrpc.Version, Error: jsonrpcErr, ID: reqID}
	if err := json.NewEncoder(rw).Encode(resp); err != nil {
		log.Error(ctx, "failed to send error response", err)
	}
}
