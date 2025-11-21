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
	"iter"
	"log/slog"

	"github.com/google/uuid"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/log"
)

// InterceptedHandler implements [RequestHandler]. It can be used to attach call interceptors and initialize
// call context for every method of the wrapped handler.
type InterceptedHandler struct {
	// Handler is responsible for the actual processing of every call.
	Handler RequestHandler
	// Interceptors is a list of call interceptors which will be applied before and after each call.
	Interceptors []CallInterceptor
	// Logger is the logger which will be accessible from request scope context using log package
	// methods. Defaults to slog.Default() if not set.
	Logger *slog.Logger
}

func (h *InterceptedHandler) OnGetTask(ctx context.Context, query *a2a.TaskQueryParams) (*a2a.Task, error) {
	ctx, callCtx := withMethodCallContext(ctx, "OnGetTask")
	if query != nil {
		ctx = h.withLoggerContext(ctx, slog.String("task_id", string(query.ID)))
	}
	ctx, err := h.interceptBefore(ctx, callCtx, query)
	if err != nil {
		return nil, err
	}
	response, err := h.Handler.OnGetTask(ctx, query)
	if errOverride := h.interceptAfter(ctx, callCtx, response, err); errOverride != nil {
		return nil, errOverride
	}
	return response, err
}

func (h *InterceptedHandler) OnCancelTask(ctx context.Context, params *a2a.TaskIDParams) (*a2a.Task, error) {
	ctx, callCtx := withMethodCallContext(ctx, "OnCancelTask")
	if params != nil {
		ctx = h.withLoggerContext(ctx, slog.String("task_id", string(params.ID)))
	}
	ctx, err := h.interceptBefore(ctx, callCtx, params)
	if err != nil {
		return nil, err
	}
	response, err := h.Handler.OnCancelTask(ctx, params)
	if errOverride := h.interceptAfter(ctx, callCtx, response, err); errOverride != nil {
		return nil, errOverride
	}
	return response, err
}

func (h *InterceptedHandler) OnSendMessage(ctx context.Context, params *a2a.MessageSendParams) (a2a.SendMessageResult, error) {
	ctx, callCtx := withMethodCallContext(ctx, "OnSendMessage")
	if params != nil && params.Message != nil {
		msg := params.Message
		ctx = h.withLoggerContext(
			ctx,
			slog.String("message_id", msg.ID),
			slog.String("task_id", string(msg.TaskID)),
			slog.String("context_id", msg.ContextID),
		)
	} else {
		ctx = h.withLoggerContext(ctx)
	}
	ctx, err := h.interceptBefore(ctx, callCtx, params)
	if err != nil {
		return nil, err
	}
	response, err := h.Handler.OnSendMessage(ctx, params)
	if errOverride := h.interceptAfter(ctx, callCtx, response, err); errOverride != nil {
		return nil, errOverride
	}
	return response, err
}

func (h *InterceptedHandler) OnSendMessageStream(ctx context.Context, params *a2a.MessageSendParams) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		ctx, callCtx := withMethodCallContext(ctx, "OnSendMessageStream")
		if params != nil && params.Message != nil {
			msg := params.Message
			ctx = h.withLoggerContext(
				ctx,
				slog.String("message_id", msg.ID),
				slog.String("task_id", string(msg.TaskID)),
				slog.String("context_id", msg.ContextID),
			)
		} else {
			ctx = h.withLoggerContext(ctx)
		}
		ctx, err := h.interceptBefore(ctx, callCtx, params)
		if err != nil {
			yield(nil, err)
			return
		}
		for event, err := range h.Handler.OnSendMessageStream(ctx, params) {
			if errOverride := h.interceptAfter(ctx, callCtx, event, err); errOverride != nil {
				yield(nil, errOverride)
				return
			}
			if !yield(event, err) {
				return
			}
		}
	}
}

func (h *InterceptedHandler) OnResubscribeToTask(ctx context.Context, params *a2a.TaskIDParams) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		ctx, callCtx := withMethodCallContext(ctx, "OnResubscribeToTask")
		if params != nil {
			ctx = h.withLoggerContext(ctx, slog.String("task_id", string(params.ID)))
		}
		ctx, err := h.interceptBefore(ctx, callCtx, params)
		if err != nil {
			yield(nil, err)
			return
		}
		for event, err := range h.Handler.OnResubscribeToTask(ctx, params) {
			if errOverride := h.interceptAfter(ctx, callCtx, event, err); errOverride != nil {
				yield(nil, errOverride)
				return
			}
			if !yield(event, err) {
				return
			}
		}
	}
}

func (h *InterceptedHandler) OnGetTaskPushConfig(ctx context.Context, params *a2a.GetTaskPushConfigParams) (*a2a.TaskPushConfig, error) {
	ctx, callCtx := withMethodCallContext(ctx, "OnGetTaskPushConfig")
	if params != nil {
		ctx = h.withLoggerContext(ctx, slog.String("task_id", string(params.TaskID)))
	}
	ctx, err := h.interceptBefore(ctx, callCtx, params)
	if err != nil {
		return nil, err
	}
	response, err := h.Handler.OnGetTaskPushConfig(ctx, params)
	if errOverride := h.interceptAfter(ctx, callCtx, response, err); errOverride != nil {
		return nil, errOverride
	}
	return response, err
}

func (h *InterceptedHandler) OnListTaskPushConfig(ctx context.Context, params *a2a.ListTaskPushConfigParams) ([]*a2a.TaskPushConfig, error) {
	ctx, callCtx := withMethodCallContext(ctx, "OnListTaskPushConfig")
	if params != nil {
		ctx = h.withLoggerContext(ctx, slog.String("task_id", string(params.TaskID)))
	}
	ctx, err := h.interceptBefore(ctx, callCtx, params)
	if err != nil {
		return nil, err
	}
	response, err := h.Handler.OnListTaskPushConfig(ctx, params)
	if errOverride := h.interceptAfter(ctx, callCtx, response, err); errOverride != nil {
		return nil, errOverride
	}
	return response, err
}

func (h *InterceptedHandler) OnSetTaskPushConfig(ctx context.Context, params *a2a.TaskPushConfig) (*a2a.TaskPushConfig, error) {
	ctx, callCtx := withMethodCallContext(ctx, "OnSetTaskPushConfig")
	if params != nil {
		ctx = h.withLoggerContext(ctx, slog.String("task_id", string(params.TaskID)))
	}
	ctx, err := h.interceptBefore(ctx, callCtx, params)
	if err != nil {
		return nil, err
	}
	response, err := h.Handler.OnSetTaskPushConfig(ctx, params)
	if errOverride := h.interceptAfter(ctx, callCtx, response, err); errOverride != nil {
		return nil, errOverride
	}
	return response, err
}

func (h *InterceptedHandler) OnDeleteTaskPushConfig(ctx context.Context, params *a2a.DeleteTaskPushConfigParams) error {
	ctx, callCtx := withMethodCallContext(ctx, "OnDeleteTaskPushConfig")
	if params != nil {
		ctx = h.withLoggerContext(ctx, slog.String("task_id", string(params.TaskID)))
	}
	ctx, err := h.interceptBefore(ctx, callCtx, params)
	if err != nil {
		return err
	}
	err = h.Handler.OnDeleteTaskPushConfig(ctx, params)
	if errOverride := h.interceptAfter(ctx, callCtx, nil, err); errOverride != nil {
		return errOverride
	}
	return err
}

func (h *InterceptedHandler) OnGetExtendedAgentCard(ctx context.Context) (*a2a.AgentCard, error) {
	ctx, callCtx := withMethodCallContext(ctx, "OnGetExtendedAgentCard")
	ctx = h.withLoggerContext(ctx)
	ctx, err := h.interceptBefore(ctx, callCtx, nil)
	if err != nil {
		return nil, err
	}
	response, err := h.Handler.OnGetExtendedAgentCard(ctx)
	if errOverride := h.interceptAfter(ctx, callCtx, response, err); errOverride != nil {
		return nil, errOverride
	}
	return response, err
}

func (h *InterceptedHandler) interceptBefore(ctx context.Context, callCtx *CallContext, payload any) (context.Context, error) {
	request := &Request{Payload: payload}

	for _, interceptor := range h.Interceptors {
		localCtx, err := interceptor.Before(ctx, callCtx, request)
		if err != nil {
			return ctx, err
		}
		ctx = localCtx
	}

	return ctx, nil
}

func (h *InterceptedHandler) interceptAfter(ctx context.Context, callCtx *CallContext, payload any, responseErr error) error {
	response := &Response{Payload: payload, Err: responseErr}

	for i := range len(h.Interceptors) {
		interceptor := h.Interceptors[len(h.Interceptors)-i-1]
		if err := interceptor.After(ctx, callCtx, response); err != nil {
			return err
		}
	}

	return nil
}

// withLoggerContext is a private utility function which attaches an slog.Logger with a2a-specific attributes
// to the provided context.
func (h *InterceptedHandler) withLoggerContext(ctx context.Context, attrs ...any) context.Context {
	logger := h.Logger
	if logger == nil {
		logger = log.LoggerFrom(ctx)
	}
	requestID := uuid.NewString()
	withAttrs := logger.WithGroup("a2a").With(attrs...).With(slog.String("request_id", requestID))
	return log.WithLogger(ctx, withAttrs)
}

// withMethodCallContext is a private utility function which modifies CallContext.method if a CallContext
// was passed by a transport implementation or initializes a new CallContext with the provided method.
func withMethodCallContext(ctx context.Context, method string) (context.Context, *CallContext) {
	callCtx, ok := CallContextFrom(ctx)
	if !ok {
		ctx, callCtx = WithCallContext(ctx, nil)
	}
	callCtx.method = method
	return ctx, callCtx
}
