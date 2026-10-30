// Copyright 2026 Google LLC
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

package agentengine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"

	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"google.golang.org/genai"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
)

// DefaultClassMethod is the streaming method every ADK agent on Agent Runtime
// exposes, in Go (server/agentengine) and in Python (AdkApp).
const DefaultClassMethod = "async_stream_query"

// StreamQueryRequest is one conversational turn.
type StreamQueryRequest struct {
	UserID    string
	SessionID string
	Message   *genai.Content
	// ClassMethod defaults to DefaultClassMethod.
	ClassMethod string
}

// StreamQuery returns an iterator that runs one turn against the reasoning
// engine and yields its events in order. The RPC is issued when the iterator
// is ranged, not when StreamQuery is called: req.Message is read at that
// point, and ranging the same iterator again runs another turn. The iterator
// ends when the server closes the stream, when a remote error line arrives
// (reported as an ErrRemote error), or when ctx is done. A done ctx is
// reported as a gRPC status error with code Canceled or DeadlineExceeded,
// and nothing received earlier is yielded after it. A ctx that ends as the
// server closes the stream may or may not be reported, depending on which
// arrives first. A server can end the RPC with those codes too, so check
// ctx.Err() to tell the caller's own cancellation from a remote one.
// Against an adk-go agent the session is created server-side on first use
// with the given SessionID (the server runs with AutoCreateSession). Other
// agent frameworks may require the session to exist first.
func (c *Client) StreamQuery(ctx context.Context, req StreamQueryRequest) iter.Seq2[*Event, error] {
	return func(yield func(*Event, error) bool) {
		// An abandoned GAPIC stream stays open until its ctx ends.
		streamCtx, cancel := context.WithCancel(ctx)
		defer cancel()
		pbReq, err := encodeRequest(c.name, req)
		if err != nil {
			yield(nil, err)
			return
		}
		stream, err := c.exec.StreamQueryReasoningEngine(streamCtx, pbReq)
		if err != nil {
			yield(nil, fmt.Errorf("agentengine: stream query %s: %w", c.name, err))
			return
		}
		chunks := func(yield func([]byte, error) bool) {
			for {
				body, err := stream.Recv()
				if errors.Is(err, io.EOF) {
					return
				}
				if err != nil {
					yield(nil, fmt.Errorf("agentengine: stream query %s: %w", c.name, err))
					return
				}
				if !yield(body.GetData(), nil) {
					return
				}
			}
		}
		for ev, err := range decodeJSONL(chunks) {
			// Lines already buffered from the last chunk would otherwise be
			// delivered after the caller gave up. Checked on ctx rather than
			// streamCtx: a derived ctx learns of a non-stdlib parent's end
			// through a goroutine and can lag it. Shaped like the error Recv
			// returns for the same cause, so callers check one thing.
			if cerr := ctx.Err(); cerr != nil {
				yield(nil, fmt.Errorf("agentengine: stream query %s: %w", c.name, status.FromContextError(cerr).Err()))
				return
			}
			if !yield(ev, err) {
				return
			}
		}
	}
}

func encodeRequest(name string, req StreamQueryRequest) (*aiplatformpb.StreamQueryReasoningEngineRequest, error) {
	if req.Message == nil {
		return nil, errors.New("agentengine: StreamQueryRequest.Message is required")
	}
	// Round-trip through JSON so the message has exactly the shape the server
	// decodes into genai.Content.
	b, err := json.Marshal(req.Message)
	if err != nil {
		return nil, fmt.Errorf("agentengine: encoding message: %w", err)
	}
	var msg map[string]any
	if err := json.Unmarshal(b, &msg); err != nil {
		return nil, fmt.Errorf("agentengine: re-decoding message: %w", err)
	}
	input, err := structpb.NewStruct(map[string]any{
		"user_id":    req.UserID,
		"session_id": req.SessionID,
		"message":    msg,
	})
	if err != nil {
		return nil, fmt.Errorf("agentengine: building input: %w", err)
	}
	method := req.ClassMethod
	if method == "" {
		method = DefaultClassMethod
	}
	return &aiplatformpb.StreamQueryReasoningEngineRequest{Name: name, ClassMethod: method, Input: input}, nil
}
