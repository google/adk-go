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

package a2aclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/internal/jsonrpc"
	"github.com/a2aproject/a2a-go/internal/sse"
	"github.com/a2aproject/a2a-go/log"
	"github.com/google/uuid"
)

// jsonrpcRequest represents a JSON-RPC 2.0 request.
type jsonrpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
	ID      string `json:"id"`
}

// jsonrpcResponse represents a JSON-RPC 2.0 response.
type jsonrpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      string          `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *jsonrpc.Error  `json:"error,omitempty"`
}

// JSONRPCOption configures optional parameters for the JSONRPC transport.
// Options are applied during NewJSONRPCTransport initialization.
type JSONRPCOption func(*jsonrpcTransport)

// WithJSONRPCTransport returns a Client factory option that enables JSON-RPC transport support.
// When applied, the client will use JSON-RPC 2.0 over HTTP for all A2A protocol communication
// as defined in the A2A specification §7.
func WithJSONRPCTransport(client *http.Client) FactoryOption {
	return WithTransport(
		a2a.TransportProtocolJSONRPC,
		TransportFactoryFn(func(ctx context.Context, url string, card *a2a.AgentCard) (Transport, error) {
			return NewJSONRPCTransport(url, client), nil
		}),
	)
}

// NewJSONRPCTransport creates a new JSON-RPC transport for A2A protocol communication.
// By default, an HTTP client with 5-second timeout is used.
// For production deployments, provide a client with appropriate timeout, retry policy,
// and connection pooling configured for your requirements.
func NewJSONRPCTransport(url string, client *http.Client) Transport {
	t := &jsonrpcTransport{
		url:        url,
		httpClient: client,
	}

	if t.httpClient == nil {
		t.httpClient = &http.Client{
			Timeout: 5 * time.Second, // Match Python SDK httpx.AsyncClient default
		}
	}

	return t
}

// jsonrpcTransport implements Transport using JSON-RPC 2.0 over HTTP.
type jsonrpcTransport struct {
	url        string
	httpClient *http.Client
}

func (t *jsonrpcTransport) newHTTPRequest(ctx context.Context, method string, params any) (*http.Request, error) {
	req := jsonrpcRequest{
		JSONRPC: jsonrpc.Version,
		Method:  method,
		Params:  params,
		ID:      uuid.New().String(),
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", t.url, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	httpReq.Header.Set("Content-Type", jsonrpc.ContentJSON)

	if callMeta, ok := CallMetaFrom(ctx); ok {
		for k, vals := range callMeta {
			for _, v := range vals {
				httpReq.Header.Add(k, v)
			}
		}
	}

	return httpReq, nil
}

// sendRequest sends a non-streaming JSON-RPC request and returns the response.
func (t *jsonrpcTransport) sendRequest(ctx context.Context, method string, params any) (json.RawMessage, error) {
	httpReq, err := t.newHTTPRequest(ctx, method, params)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	httpResp, err := t.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send HTTP request: %w", err)
	}
	defer func() {
		if err := httpResp.Body.Close(); err != nil {
			log.Error(ctx, "failed to close http response body", err)
		}
	}()

	if httpResp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected HTTP status: %s", httpResp.Status)
	}

	var resp jsonrpcResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if resp.Error != nil {
		return nil, resp.Error.ToA2AError()
	}

	return resp.Result, nil
}

// sendStreamingRequest sends a streaming JSON-RPC request and returns an SSE stream.
func (t *jsonrpcTransport) sendStreamingRequest(ctx context.Context, method string, params any) (io.ReadCloser, error) {
	httpReq, err := t.newHTTPRequest(ctx, method, params)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}
	httpReq.Header.Set("Accept", sse.ContentEventStream)

	httpResp, err := t.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send HTTP request: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		if err := httpResp.Body.Close(); err != nil {
			log.Error(ctx, "failed to close http response body", err)
		}
		return nil, fmt.Errorf("unexpected HTTP status: %s", httpResp.Status)
	}

	return httpResp.Body, nil
}

// parseSSEStream parses Server-Sent Events and yields JSON-RPC responses.
func parseSSEStream(body io.Reader) iter.Seq2[json.RawMessage, error] {
	return func(yield func(json.RawMessage, error) bool) {
		for data, err := range sse.ParseDataStream(body) {
			if err != nil {
				yield(nil, err)
				return
			}
			var resp jsonrpcResponse
			if err := json.Unmarshal(data, &resp); err != nil {
				yield(nil, fmt.Errorf("failed to parse SSE data: %w", err))
				return
			}
			if resp.Error != nil {
				yield(nil, resp.Error.ToA2AError())
				return
			}
			if !yield(resp.Result, nil) {
				return
			}
		}
	}
}

// SendMessage sends a non-streaming message to the agent.
func (t *jsonrpcTransport) SendMessage(ctx context.Context, message *a2a.MessageSendParams) (a2a.SendMessageResult, error) {
	result, err := t.sendRequest(ctx, jsonrpc.MethodMessageSend, message)
	if err != nil {
		return nil, err
	}

	// Use a2a.UnmarshalEventJSON to determine the type based on the 'kind' field
	event, err := a2a.UnmarshalEventJSON(result)
	if err != nil {
		return nil, fmt.Errorf("result violates A2A spec - could not determine type: %w; data: %s", err, string(result))
	}

	// SendMessage can return either a Task or a Message
	switch e := event.(type) {
	case *a2a.Task:
		return e, nil
	case *a2a.Message:
		return e, nil
	default:
		return nil, fmt.Errorf("result violates A2A spec - expected Task or Message, got %T: %s", event, string(result))
	}
}

// streamRequestToEvents handles SSE streaming for JSON-RPC methods.
// It converts the SSE stream into a sequence of A2A events.
func (t *jsonrpcTransport) streamRequestToEvents(ctx context.Context, method string, params any) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		body, err := t.sendStreamingRequest(ctx, method, params)
		if err != nil {
			yield(nil, err)
			return
		}
		defer func() {
			if err := body.Close(); err != nil {
				log.Error(ctx, "failed to close http response body", err)
			}
		}()

		for result, err := range parseSSEStream(body) {
			if err != nil {
				yield(nil, err)
				return
			}

			event, err := a2a.UnmarshalEventJSON(result)
			if err != nil {
				yield(nil, err)
				return
			}

			if !yield(event, nil) {
				return
			}
		}
	}
}

// SendStreamingMessage sends a streaming message to the agent.
func (t *jsonrpcTransport) SendStreamingMessage(ctx context.Context, message *a2a.MessageSendParams) iter.Seq2[a2a.Event, error] {
	return t.streamRequestToEvents(ctx, jsonrpc.MethodMessageStream, message)
}

// GetTask retrieves the current state of a task.
func (t *jsonrpcTransport) GetTask(ctx context.Context, query *a2a.TaskQueryParams) (*a2a.Task, error) {
	result, err := t.sendRequest(ctx, jsonrpc.MethodTasksGet, query)
	if err != nil {
		return nil, err
	}

	var task a2a.Task
	if err := json.Unmarshal(result, &task); err != nil {
		return nil, fmt.Errorf("failed to unmarshal task: %w", err)
	}

	return &task, nil
}

// CancelTask requests cancellation of a task.
func (t *jsonrpcTransport) CancelTask(ctx context.Context, id *a2a.TaskIDParams) (*a2a.Task, error) {
	result, err := t.sendRequest(ctx, jsonrpc.MethodTasksCancel, id)
	if err != nil {
		return nil, err
	}

	var task a2a.Task
	if err := json.Unmarshal(result, &task); err != nil {
		return nil, fmt.Errorf("failed to unmarshal task: %w", err)
	}

	return &task, nil
}

// ResubscribeToTask reconnects to an SSE stream for an ongoing task.
func (t *jsonrpcTransport) ResubscribeToTask(ctx context.Context, id *a2a.TaskIDParams) iter.Seq2[a2a.Event, error] {
	return t.streamRequestToEvents(ctx, jsonrpc.MethodTasksResubscribe, id)
}

// GetTaskPushConfig retrieves the push notification configuration for a task.
func (t *jsonrpcTransport) GetTaskPushConfig(ctx context.Context, params *a2a.GetTaskPushConfigParams) (*a2a.TaskPushConfig, error) {
	result, err := t.sendRequest(ctx, jsonrpc.MethodPushConfigGet, params)
	if err != nil {
		return nil, err
	}

	var config a2a.TaskPushConfig
	if err := json.Unmarshal(result, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &config, nil
}

// ListTaskPushConfig lists push notification configurations.
func (t *jsonrpcTransport) ListTaskPushConfig(ctx context.Context, params *a2a.ListTaskPushConfigParams) ([]*a2a.TaskPushConfig, error) {
	result, err := t.sendRequest(ctx, jsonrpc.MethodPushConfigList, params)
	if err != nil {
		return nil, err
	}

	var configs []*a2a.TaskPushConfig
	if err := json.Unmarshal(result, &configs); err != nil {
		return nil, fmt.Errorf("failed to unmarshal configs: %w", err)
	}

	return configs, nil
}

// SetTaskPushConfig sets or updates the push notification configuration for a task.
func (t *jsonrpcTransport) SetTaskPushConfig(ctx context.Context, params *a2a.TaskPushConfig) (*a2a.TaskPushConfig, error) {
	result, err := t.sendRequest(ctx, jsonrpc.MethodPushConfigSet, params)
	if err != nil {
		return nil, err
	}

	var config a2a.TaskPushConfig
	if err := json.Unmarshal(result, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &config, nil
}

// DeleteTaskPushConfig deletes a push notification configuration.
func (t *jsonrpcTransport) DeleteTaskPushConfig(ctx context.Context, params *a2a.DeleteTaskPushConfigParams) error {
	_, err := t.sendRequest(ctx, jsonrpc.MethodPushConfigDelete, params)
	return err
}

// GetAgentCard retrieves the agent's card.
func (t *jsonrpcTransport) GetAgentCard(ctx context.Context) (*a2a.AgentCard, error) {
	result, err := t.sendRequest(ctx, jsonrpc.MethodGetExtendedAgentCard, nil)
	if err != nil {
		return nil, err
	}

	var card a2a.AgentCard
	if err := json.Unmarshal(result, &card); err != nil {
		return nil, fmt.Errorf("failed to unmarshal agent card: %w", err)
	}
	return &card, nil
}

// Destroy closes the transport and releases resources.
func (t *jsonrpcTransport) Destroy() error {
	// HTTP client doesn't need explicit cleanup in most cases
	// If a custom client with cleanup is needed, implement via options
	return nil
}
