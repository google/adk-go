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

package functiontool_test

import (
	"errors"
	"fmt"
	"iter"
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

type streamingArgs struct {
	Value string `json:"value"`
}

func newStreamingToolContext(t *testing.T, actions *session.EventActions, confirmation *toolconfirmation.ToolConfirmation) agent.Context {
	t.Helper()
	invocationContext := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	return agent.NewToolContext(invocationContext, "call-id", actions, confirmation)
}

func requireStreamingTool(t *testing.T, cfg functiontool.Config, handler functiontool.StreamingFunc[streamingArgs]) toolinternal.StreamingFunctionTool {
	t.Helper()
	got, err := functiontool.NewStreaming(cfg, handler)
	if err != nil {
		t.Fatalf("NewStreaming() failed: %v", err)
	}
	streamingTool, ok := got.(toolinternal.StreamingFunctionTool)
	if !ok {
		t.Fatalf("NewStreaming() returned %T, want toolinternal.StreamingFunctionTool", got)
	}
	return streamingTool
}

func TestStreamingFunctionToolConfirmation(t *testing.T) {
	tests := []struct {
		name                 string
		confirmation         *toolconfirmation.ToolConfirmation
		wantErr              error
		wantHandlerCall      bool
		wantConfirmationCall bool
	}{
		{
			name:         "rejected",
			confirmation: &toolconfirmation.ToolConfirmation{Confirmed: false},
			wantErr:      tool.ErrConfirmationRejected,
		},
		{
			name:            "confirmed",
			confirmation:    &toolconfirmation.ToolConfirmation{Confirmed: true},
			wantHandlerCall: true,
		},
		{
			name:                 "confirmation requested",
			wantErr:              tool.ErrConfirmationRequired,
			wantConfirmationCall: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actions := &session.EventActions{}
			handlerCalled := false
			streamingTool := requireStreamingTool(t, functiontool.Config{
				Name:                "streaming_tool",
				RequireConfirmation: true,
			}, func(_ agent.Context, args streamingArgs) iter.Seq2[string, error] {
				handlerCalled = true
				return func(yield func(string, error) bool) {
					yield("handled "+args.Value, nil)
				}
			})

			var values []string
			var gotErr error
			ctx := newStreamingToolContext(t, actions, test.confirmation)
			for value, err := range streamingTool.RunStream(ctx, map[string]any{"value": "request"}) {
				values = append(values, value)
				gotErr = err
			}

			if !errors.Is(gotErr, test.wantErr) {
				t.Errorf("RunStream() error = %v, want %v", gotErr, test.wantErr)
			}
			if handlerCalled != test.wantHandlerCall {
				t.Errorf("handler called = %t, want %t", handlerCalled, test.wantHandlerCall)
			}
			if test.wantHandlerCall {
				if len(values) != 1 || values[0] != "handled request" {
					t.Errorf("RunStream() values = %v, want [handled request]", values)
				}
			}

			confirmation, requested := actions.RequestedToolConfirmations["call-id"]
			if requested != test.wantConfirmationCall {
				t.Fatalf("confirmation requested = %t, want %t", requested, test.wantConfirmationCall)
			}
			if test.wantConfirmationCall {
				if !actions.SkipSummarization {
					t.Error("SkipSummarization = false, want true")
				}
				if !strings.Contains(confirmation.Hint, "streaming_tool()") {
					t.Errorf("confirmation hint = %q, want tool name", confirmation.Hint)
				}
			}
		})
	}
}

func TestNewStreamingRequireConfirmationProviderValidation(t *testing.T) {
	tests := []struct {
		name     string
		provider any
		wantErr  bool
	}{
		{name: "nil", provider: nil},
		{name: "valid", provider: func(streamingArgs) bool { return true }},
		{name: "not a function", provider: struct{}{}, wantErr: true},
		{name: "wrong argument", provider: func(string) bool { return true }, wantErr: true},
		{name: "wrong result", provider: func(streamingArgs) int { return 1 }, wantErr: true},
	}

	handler := func(agent.Context, streamingArgs) iter.Seq2[string, error] {
		return func(func(string, error) bool) {}
	}
	wantError := fmt.Sprintf("error RequireConfirmationProvider must be a function with signature func(%T) bool", streamingArgs{})
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := functiontool.NewStreaming[streamingArgs](functiontool.Config{
				RequireConfirmationProvider: test.provider,
			}, handler)
			if test.wantErr {
				if err == nil || !strings.Contains(err.Error(), wantError) {
					t.Fatalf("NewStreaming() error = %v, want error containing %q", err, wantError)
				}
				return
			}
			if err != nil {
				t.Fatalf("NewStreaming() failed: %v", err)
			}
			if got == nil {
				t.Fatal("NewStreaming() returned nil tool")
			}
		})
	}
}

func TestStreamingFunctionToolStopsHandlerWhenConsumerStops(t *testing.T) {
	cleanedUp := false
	streamingTool := requireStreamingTool(t, functiontool.Config{Name: "streaming_tool"}, func(_ agent.Context, _ streamingArgs) iter.Seq2[string, error] {
		return func(yield func(string, error) bool) {
			defer func() { cleanedUp = true }()
			if !yield("first", nil) {
				return
			}
			yield("second", nil)
		}
	})

	ctx := newStreamingToolContext(t, &session.EventActions{}, nil)
	for range streamingTool.RunStream(ctx, map[string]any{"value": "request"}) {
		break
	}

	if !cleanedUp {
		t.Error("handler cleanup did not run after the consumer stopped")
	}
}

type streamingConfirmationFailureContext struct {
	agent.Context
}

func (streamingConfirmationFailureContext) RequestConfirmation(string, any) error {
	return errors.New("synthetic confirmation failure")
}

func captureStreamingPanic(f func()) (got any) {
	defer func() { got = recover() }()
	f()
	return got
}

func TestStreamingFunctionToolPropagatesConsumerPanic(t *testing.T) {
	validArgs := map[string]any{"value": "request"}
	for _, test := range []struct {
		name                string
		args                any
		confirmation        *toolconfirmation.ToolConfirmation
		requireConfirmation bool
		confirmationFailure bool
		handlerError        bool
		wantError           bool
	}{
		{name: "chunk", args: validArgs},
		{name: "handler error", args: validArgs, handlerError: true, wantError: true},
		{name: "wrong argument type", args: 42, wantError: true},
		{name: "schema error", args: map[string]any{"value": false}, wantError: true},
		{name: "confirmation rejected", args: validArgs, confirmation: &toolconfirmation.ToolConfirmation{}, wantError: true},
		{name: "confirmation requested", args: validArgs, requireConfirmation: true, wantError: true},
		{name: "confirmation request failed", args: validArgs, requireConfirmation: true, confirmationFailure: true, wantError: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			streamingTool := requireStreamingTool(t, functiontool.Config{
				Name:                "streaming_tool",
				RequireConfirmation: test.requireConfirmation,
			}, func(agent.Context, streamingArgs) iter.Seq2[string, error] {
				return func(yield func(string, error) bool) {
					var err error
					if test.handlerError {
						err = errors.New("synthetic handler error")
					}
					yield("chunk", err)
				}
			})
			ctx := newStreamingToolContext(t, &session.EventActions{}, test.confirmation)
			if test.confirmationFailure {
				ctx = streamingConfirmationFailureContext{Context: ctx}
			}
			original := errors.New("synthetic consumer panic")
			var yieldedErr error
			got := captureStreamingPanic(func() {
				for _, err := range streamingTool.RunStream(ctx, test.args) {
					yieldedErr = err
					panic(original)
				}
			})
			if got != original {
				t.Errorf("recovered panic has type %T, want the original consumer panic value", got)
			}
			if (yieldedErr != nil) != test.wantError {
				t.Errorf("yielded an error = %t, want %t", yieldedErr != nil, test.wantError)
			}
		})
	}
}

func TestStreamingFunctionToolPropagatesCleanupPanicAfterStop(t *testing.T) {
	original := errors.New("synthetic cleanup panic")
	streamingTool := requireStreamingTool(t, functiontool.Config{Name: "streaming_tool"}, func(agent.Context, streamingArgs) iter.Seq2[string, error] {
		return func(yield func(string, error) bool) {
			defer func() { panic(original) }()
			yield("chunk", nil)
		}
	})
	ctx := newStreamingToolContext(t, &session.EventActions{}, nil)
	got := captureStreamingPanic(func() {
		for range streamingTool.RunStream(ctx, map[string]any{"value": "request"}) {
			break
		}
	})
	if got != original {
		t.Errorf("recovered panic has type %T, want the original cleanup panic value", got)
	}
}

func TestStreamingFunctionToolRecoversHandlerPanic(t *testing.T) {
	for _, test := range []struct {
		name        string
		panicOnCall bool
		yieldFirst  bool
	}{
		{name: "calling the handler", panicOnCall: true},
		{name: "before first yield"},
		{name: "after a yield", yieldFirst: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			streamingTool := requireStreamingTool(t, functiontool.Config{Name: "streaming_tool"}, func(agent.Context, streamingArgs) iter.Seq2[string, error] {
				if test.panicOnCall {
					panic("synthetic handler panic")
				}
				return func(yield func(string, error) bool) {
					if test.yieldFirst && !yield("chunk", nil) {
						return
					}
					panic("synthetic handler panic")
				}
			})
			ctx := newStreamingToolContext(t, &session.EventActions{}, nil)
			var chunks, errs int
			got := captureStreamingPanic(func() {
				for _, err := range streamingTool.RunStream(ctx, map[string]any{"value": "request"}) {
					if err == nil {
						chunks++
						continue
					}
					errs++
					if !strings.Contains(err.Error(), "panic in tool") {
						t.Error("RunStream did not report the recovered handler panic")
					}
				}
			})
			if got != nil {
				t.Fatalf("RunStream propagated a panic of type %T, want a tool error", got)
			}
			wantChunks := 0
			if test.yieldFirst {
				wantChunks = 1
			}
			if chunks != wantChunks || errs != 1 {
				t.Errorf("RunStream yielded %d chunks and %d errors, want %d chunks and one error", chunks, errs, wantChunks)
			}
		})
	}
}
