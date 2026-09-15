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

package llminternal

import (
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"reflect"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// basicRequestProcessor populates the LLMRequest
// with the agent's LLM generation configs.
func basicRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	// reference: adk-python src/google/adk/flows/llm_flows/basic.py
	return func(yield func(*session.Event, error) bool) {
		llmAgent := asLLMAgent(ctx.Agent())
		if llmAgent == nil {
			return // do nothing.
		}

		state := llmAgent.internal()

		config, err := clone(state.GenerateContentConfig)
		if err != nil {
			yield(nil, fmt.Errorf("clone generation config: %w", err))
			return
		}
		req.Config = config
		if req.Config == nil {
			req.Config = &genai.GenerateContentConfig{}
		}

		// Set OutputSchema directly if no tools are present or native combo support exists.
		// Otherwise, OutputSchemaRequestProcessor will be used to provide a tool-based workaround.
		//
		// Task-mode agents skip OutputSchema configuration entirely:
		// structured output for tasks is collected via the FinishTaskTool's
		// declaration (the model emits the value inside the finish_task
		// FC args, not as a structured text response).
		if state.Mode != ModeTask && state.OutputSchema != nil && !needOutputSchemaProcessor(state) {
			req.Config.ResponseSchema = state.OutputSchema
			req.Config.ResponseMIMEType = "application/json"
		}

		// TODO: missing features
		//  populate LLMRequest LiveConnectConfig setting
	}
}

// clone copies src's data containers, with time.Time values copied by value.
// Inside an empty interface, values with unexported fields are preserved as an
// independent json.RawMessage instead of retaining references to private state.
// Other concrete types are preserved. Functions, channels, and unsafe pointers
// are retained as-is. Unsupported types and excessive nesting return an error.
func clone[M any](src M) (M, error) {
	var dst M
	if err := deepCopy(reflect.ValueOf(&src).Elem(), reflect.ValueOf(&dst).Elem(), 0); err != nil {
		var zero M
		return zero, err
	}
	return dst, nil
}

// maxCloneDepth counts reflection steps, including pointers and interfaces,
// to bound recursion through cyclic or excessively nested payloads.
const maxCloneDepth = 1024

var (
	errCloneDepth      = errors.New("maximum clone depth exceeded")
	errCloneUnexported = errors.New("cannot clone unexported field")
)

// deepCopy copies src to dst using reflect.
func deepCopy(src, dst reflect.Value, depth int) error {
	if depth > maxCloneDepth {
		return fmt.Errorf("%w: limit %d (cyclic or excessively nested value)", errCloneDepth, maxCloneDepth)
	}
	switch src.Kind() {
	case reflect.Struct:
		t := src.Type()
		if t == reflect.TypeFor[time.Time]() {
			// time.Time's API supports value copying, including its location data.
			dst.Set(src)
			return nil
		}
		for i := 0; i < src.NumField(); i++ {
			if !t.Field(i).IsExported() {
				return fmt.Errorf("%w %q in type %v", errCloneUnexported, t.Field(i).Name, t)
			}
			// Create a copy of the field and set it on the destination struct
			fieldCopy := reflect.New(src.Field(i).Type()).Elem()
			if err := deepCopy(src.Field(i), fieldCopy, depth+1); err != nil {
				return err
			}
			dst.Field(i).Set(fieldCopy)
		}
	case reflect.Interface:
		if src.IsNil() {
			return nil
		}
		valueCopy := reflect.New(src.Elem().Type()).Elem()
		if err := deepCopy(src.Elem(), valueCopy, depth+1); err != nil {
			if !errors.Is(err, errCloneUnexported) || src.Type().NumMethod() != 0 {
				return err
			}
			// JSON preserves opaque payloads (including custom marshalers) without
			// exposing private state or rounding numbers through float64.
			raw, err := json.Marshal(src.Interface())
			if err != nil {
				return fmt.Errorf("clone JSON value of type %v: %w", src.Elem().Type(), err)
			}
			valueCopy = reflect.ValueOf(json.RawMessage(raw))
		}
		dst.Set(valueCopy)
	case reflect.Array:
		for i := 0; i < src.Len(); i++ {
			if err := deepCopy(src.Index(i), dst.Index(i), depth+1); err != nil {
				return err
			}
		}
	case reflect.Slice:
		if src.IsNil() {
			return nil
		}
		dst.Set(reflect.MakeSlice(src.Type(), src.Len(), src.Cap()))
		for i := 0; i < src.Len(); i++ {
			// Create a copy of each element and set it in the new slice
			elemCopy := reflect.New(src.Index(i).Type()).Elem()
			if err := deepCopy(src.Index(i), elemCopy, depth+1); err != nil {
				return err
			}
			dst.Index(i).Set(elemCopy)
		}
	case reflect.Map:
		if src.IsNil() {
			return nil
		}
		dst.Set(reflect.MakeMap(src.Type()))
		for _, key := range src.MapKeys() {
			// Create copies of the key and value and set them in the new map
			keyCopy := reflect.New(key.Type()).Elem()
			if err := deepCopy(key, keyCopy, depth+1); err != nil {
				return err
			}
			if !keyCopy.Comparable() {
				return fmt.Errorf("cloned map key of type %v is not comparable", key.Type())
			}
			valCopy := reflect.New(src.MapIndex(key).Type()).Elem()
			if err := deepCopy(src.MapIndex(key), valCopy, depth+1); err != nil {
				return err
			}
			dst.SetMapIndex(keyCopy, valCopy)
		}
	case reflect.Pointer:
		if src.IsNil() {
			return nil
		}
		// Create a new pointer and deep copy the underlying value
		newPtr := reflect.New(src.Elem().Type())
		if err := deepCopy(src.Elem(), newPtr.Elem(), depth+1); err != nil {
			return err
		}
		dst.Set(newPtr)
	default:
		// For basic types, direct assignment is sufficient
		dst.Set(src)
	}
	return nil
}
