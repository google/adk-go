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

// Package fallbackmodel provides a [model.LLM] that tries a list of models in
// order, and moves on to the next one when a call fails with a status such as
// 429 or 503.
//
//	m, err := fallbackmodel.New(fallbackmodel.Config{
//		Models: []model.LLM{primary, backup},
//	})
//	if err != nil {
//		return err
//	}
//	a, err := llmagent.New(llmagent.Config{Name: "reliable_agent", Model: m})
//
// Each model is tried once. Retrying one model is that model's own concern,
// for example through its client's retry options, so that one failure is not
// retried twice over by two layers.
//
// Only a failure that carries one of [Config.RetriableStatusCodes] moves on.
// Any other error reaches the caller unchanged, and so does every error once
// the caller's context is done. The status is read from the error types that
// the models in this module return; see [Config.RetriableStatusCodes].
//
// Once a model has yielded a response, or an error that does not move on, the
// turn belongs to that model. A later error reaches the caller rather than
// moving on, because the caller already holds part of that model's output and
// a second model would be spliced onto it. This matters when streaming, where a call can fail after
// several chunks. A call that does not stream yields once, so it almost always
// fails before that point and can still move on.
//
// Live sessions are not supported. The model has no Client method, so running
// an agent that uses it in live mode reports that the model does not support
// a live connection. adk-python's FallbackModel also fails over when a live
// connection is opened, but ADK Go's live flow talks to a *genai.Client
// directly, so there is no connection here to wrap.
//
// It is a port of adk-python's FallbackModel (models/_fallback_model.py).
//
// EXPERIMENTAL: this package is not yet stable API. Anything here may change or
// be removed in a later version without notice.
package fallbackmodel

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"net/http"
	"reflect"
	"slices"

	"github.com/openai/openai-go/v3"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/anthropicmodel"
)

// DefaultStatusCodes returns the HTTP status codes that move a call on to the
// next model when [Config.RetriableStatusCodes] is empty: 429, 500, 502, 503
// and 504. It returns a new slice on every call, so the result can be extended:
//
//	codes := append(fallbackmodel.DefaultStatusCodes(), anthropicmodel.StatusOverloaded)
//
// 408 is left out on purpose. A timeout does not say whether the request was
// processed before the clock ran out, so moving to another model risks paying
// for, and acting on, one prompt twice. Add it for a provider whose 408 is
// known to mean that the request was dropped.
func DefaultStatusCodes() []int {
	return []int{
		http.StatusTooManyRequests,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout,
	}
}

// Config configures the model that [New] returns.
type Config struct {
	// Models are the models to try, in order. At least one is required, and
	// none may be nil.
	//
	// The first is the primary model. The fallback model reports its name and
	// its Google backend variant, so the agent builds each request for the
	// primary. A backup is sent that same request; the request is built before
	// it is known which model will serve it.
	//
	// adk-python also accepts model names here. To build a model from a name,
	// call [model.NewLLM] first.
	Models []model.LLM

	// RetriableStatusCodes are the HTTP status codes that move a call on to
	// the next model. Empty means [DefaultStatusCodes].
	//
	// The status is read from a [genai.APIError] (model/gemini and
	// model/apigee), an [*openai.Error] (model/openaimodel) or an
	// [*anthropicmodel.APIError] (model/anthropicmodel), anywhere in the
	// error's chain. An error of any other type carries no status and never
	// moves on. A custom [model.LLM] whose failures should move on can return
	// a genai.APIError with the HTTP status in Code.
	//
	// model/anthropicmodel gives an error event that arrives in a stream, after
	// the HTTP response has succeeded, a status taken from the event's type:
	// 500 for api_error and 504 for timeout_error, for example. Such an error
	// moves on if it arrives before the stream's first response, although the
	// provider has accepted the request.
	RetriableStatusCodes []int

	// Logger receives a DEBUG record each time a call moves on to the next
	// model. The record names the model that failed, the status it failed
	// with and the model tried next, and nothing from the request or the
	// error text. Nil records nothing.
	Logger *slog.Logger
}

// New returns a model that tries each of cfg.Models in turn. It returns an
// error if cfg.Models is empty or holds a nil model.
//
// The returned model holds no per-call state, so it is safe for concurrent use
// when the models it wraps are.
func New(cfg Config) (model.LLM, error) {
	if len(cfg.Models) == 0 {
		return nil, errors.New("fallbackmodel: Config.Models is empty")
	}
	for i, m := range cfg.Models {
		if m == nil {
			return nil, fmt.Errorf("fallbackmodel: Config.Models[%d] is nil", i)
		}
	}
	codes := cfg.RetriableStatusCodes
	// Empty means the defaults, so that the zero Config works. In adk-python
	// an empty set turns falling back off instead.
	if len(codes) == 0 {
		codes = DefaultStatusCodes()
	}
	retriable := make(map[int]bool, len(codes))
	for _, code := range codes {
		retriable[code] = true
	}
	return &fallbackModel{
		models:    slices.Clone(cfg.Models),
		retriable: retriable,
		logger:    cfg.Logger,
	}, nil
}

type fallbackModel struct {
	models    []model.LLM
	retriable map[int]bool
	logger    *slog.Logger
}

var _ googlellm.GoogleLLM = (*fallbackModel)(nil)

// Name returns the primary model's name.
func (m *fallbackModel) Name() string {
	return m.models[0].Name()
}

// GetGoogleLLMVariant returns the primary model's Google backend variant, or
// genai.BackendUnspecified when the primary does not report one.
//
// The flow reads the variant while it builds the request, before any model is
// called, to decide for example whether an output schema can be sent alongside
// tools. The request is built for the primary, as adk-python's
// FallbackModel.capabilities reports the primary's capabilities.
func (m *fallbackModel) GetGoogleLLMVariant() genai.Backend {
	return googlellm.GetGoogleLLMVariant(m.models[0])
}

// GenerateContent calls each model in turn until one succeeds, and yields what
// that model yields. If the last model fails too, the caller gets its error
// unchanged.
//
// Each call sets req.Model to the name of the model being called, since the
// models read it to choose what to call. This replaces a name that a
// BeforeModelCallback put there, as adk-python does. Before a model other than
// the last is
// called, the request's Contents slice and Config are saved, and put back if
// the call fails and the next model is tried. The model that serves the call
// keeps its edits, and req.Model is left naming it.
func (m *fallbackModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		last := len(m.models) - 1
		for i, delegate := range m.models {
			// Models edit the request before sending it: Gemini appends a user
			// turn and writes headers into the config. Without a rollback, a
			// failed model would hand those edits to the next one.
			var pristine snapshot
			if i < last {
				pristine = takeSnapshot(req)
			}
			req.Model = delegate.Name()

			committed := false
			status, fallBack := 0, false
			for resp, err := range delegate.GenerateContent(ctx, req, stream) {
				if err != nil && i < last && !committed {
					if status, fallBack = m.shouldFallBack(ctx, err); fallBack {
						break
					}
				}
				// A nil response gives the caller nothing to build on, and the
				// flow skips it, so it does not commit the turn to this model.
				// adk-python commits on anything a model yields.
				committed = committed || resp != nil || err != nil
				// An error that does not move on is passed through like any
				// other value, and the model's stream goes on for as long as
				// the caller reads it: the flow reads on when an on-model-error
				// callback answers the error. A Python generator ends at its
				// first exception, so adk-python has no such case.
				if !yield(resp, err) {
					return
				}
			}
			if !fallBack {
				return
			}
			// adk-python logs this at WARNING. ADK Go logs per-invocation
			// events at DEBUG.
			if m.logger != nil {
				m.logger.DebugContext(ctx, "falling back to the next model",
					"model", delegate.Name(), "status_code", status, "next_model", m.models[i+1].Name())
			}
			pristine.restore(req)
		}
	}
}

// shouldFallBack reports whether err moves a call on to the next model, and
// the status it carries. Once ctx is done no call moves on: the caller has
// stopped waiting, and the next model would only be canceled too.
func (m *fallbackModel) shouldFallBack(ctx context.Context, err error) (int, bool) {
	if ctx.Err() != nil {
		return 0, false
	}
	status, ok := statusCode(err)
	return status, ok && m.retriable[status]
}

// statusCode returns the HTTP status carried by err, if err is or wraps one of
// the error types that this module's models return.
//
// adk-python reads a status_code attribute off any exception. Go has no
// attribute lookup, so the types are named instead.
func statusCode(err error) (int, bool) {
	if e, ok := errors.AsType[genai.APIError](err); ok {
		return e.Code, true
	}
	if e, ok := errors.AsType[*openai.Error](err); ok {
		return e.StatusCode, true
	}
	if e, ok := errors.AsType[*anthropicmodel.APIError](err); ok {
		return e.StatusCode, true
	}
	return 0, false
}

// snapshot holds the parts of a request that a model edits, so that a failed
// call can be rolled back before the next model is called.
//
// Config is copied in depth, because Gemini writes into the config's header
// map in place. Contents is copied as a slice, which undoes a model appending
// or replacing contents, but the Content values are shared: an edit made
// inside a Content is not undone. adk-python deep-copies the contents. No model
// in this module edits a Content, and copying them would copy every inline
// image and audio clip in the history before each call.
//
// Tools is not copied: it holds the live tool values, which models read and
// never edit. Model is set for each call. CacheConfig, CacheMetadata and
// PreviousPromptTokenCount are set by the flow and only read by models.
type snapshot struct {
	contents []*genai.Content
	config   *genai.GenerateContentConfig
}

func takeSnapshot(req *model.LLMRequest) snapshot {
	s := snapshot{contents: slices.Clone(req.Contents)}
	if req.Config != nil {
		s.config = deepCopy(reflect.ValueOf(req.Config), 0).Interface().(*genai.GenerateContentConfig)
	}
	return s
}

// restore puts s back on req, which the caller still holds, so it is edited
// in place rather than replaced.
func (s snapshot) restore(req *model.LLMRequest) {
	req.Contents = s.contents
	req.Config = s.config
}

// maxCopyDepth bounds deepCopy. A value deeper than this stays shared. The
// bound keeps a self-referencing value held in an interface field, such as an
// HTTPOptions.ExtraBody entry, from recursing forever. It does not limit the
// work for a cycle that branches, such as m["a"], m["b"] = m, m, which
// JSON-shaped data never contains.
const maxCopyDepth = 64

// deepCopy returns a copy of v that shares no pointer, map or slice with it,
// down to maxCopyDepth. Unexported struct fields, such as those of the
// time.Time in a Google Search time range filter, are copied by value. Funcs,
// channels and the elements of arrays, which genai types do not use, stay
// shared.
//
// The config is walked with reflection rather than field by field so that a
// field genai adds later is copied too. It follows deepCopyValue in
// artifact/clone.go.
func deepCopy(v reflect.Value, depth int) reflect.Value {
	if depth > maxCopyDepth || !mayAlias(v.Type()) {
		return v
	}
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			return v
		}
		out := reflect.New(v.Type().Elem())
		out.Elem().Set(deepCopy(v.Elem(), depth+1))
		return out
	case reflect.Interface:
		if v.IsNil() {
			return v
		}
		out := reflect.New(v.Type()).Elem()
		out.Set(deepCopy(v.Elem(), depth+1))
		return out
	case reflect.Map:
		if v.IsNil() {
			return v
		}
		out := reflect.MakeMapWithSize(v.Type(), v.Len())
		for it := v.MapRange(); it.Next(); {
			out.SetMapIndex(it.Key(), deepCopy(it.Value(), depth+1))
		}
		return out
	case reflect.Slice:
		if v.IsNil() {
			return v
		}
		out := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		reflect.Copy(out, v)
		// reflect.Copy is the whole copy for elements that hold no reference,
		// such as the bytes of inline data.
		if !mayAlias(v.Type().Elem()) {
			return out
		}
		for i := range v.Len() {
			out.Index(i).Set(deepCopy(v.Index(i), depth+1))
		}
		return out
	case reflect.Struct:
		out := reflect.New(v.Type()).Elem()
		// Assign the whole struct first: unexported fields cannot be set one
		// by one, and leaving them out would zero them.
		out.Set(v)
		for i := range v.NumField() {
			if v.Type().Field(i).IsExported() {
				out.Field(i).Set(deepCopy(v.Field(i), depth+1))
			}
		}
		return out
	default:
		return v
	}
}

// mayAlias reports whether a value of type t can hold a reference to memory
// that another holder may write through.
func mayAlias(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func, reflect.UnsafePointer:
		return true
	case reflect.Struct:
		for field := range t.Fields() {
			if mayAlias(field.Type) {
				return true
			}
		}
	}
	return false
}
