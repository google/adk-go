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
	"fmt"
	"iter"
	"reflect"

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

		req.Config = clone(state.GenerateContentConfig)
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
		if state.Mode != ModeTask {
			if schema := outputSchemaFor(ctx, state); schema != nil && !needOutputSchemaProcessor(state) {
				req.Config.ResponseSchema = schema
				req.Config.ResponseMIMEType = "application/json"
			}
		}

		// TODO: missing features
		//  populate LLMRequest LiveConnectConfig setting
	}
}

// CloneGenerateContentConfig returns a copy of cfg, or nil for nil, so callers
// can set per-request fields without mutating the shared value. It makes the
// same copy basicRequestProcessor makes for every agent request: pointers,
// slices, maps, and struct fields are copied recursively, while values held in
// interface-typed fields, such as HTTPOptions.ExtraBody entries or
// ResponseJsonSchema, remain shared with cfg. Like clone, it panics on a
// struct with unexported fields.
func CloneGenerateContentConfig(cfg *genai.GenerateContentConfig) *genai.GenerateContentConfig {
	return clone(cfg)
}

// clone returns a deep copy of the src.
// NOTE: this does not work for types with unexported fields.
func clone[M any](src M) M {
	val := reflect.ValueOf(src)

	// Handle nil pointers
	if val.Kind() == reflect.Pointer && val.IsNil() {
		var zero M
		return zero
	}

	srcIsPointer := val.Kind() == reflect.Pointer

	// Dereference pointer to get the underlying value
	if srcIsPointer {
		val = val.Elem()
	}

	// Create a new instance of the same type
	newVal := reflect.New(val.Type()).Elem()

	// Recursively copy fields
	deepCopy(val, newVal)

	// Return as the original type
	if srcIsPointer {
		return newVal.Addr().Interface().(M)
	}
	return newVal.Interface().(M)
}

// deepCopy copies src to dst using reflect. dst must be settable and hold
// either the zero value of src's type or a shallow copy of src. Structs,
// pointers, slices and maps are copied recursively; values of every other
// kind, including interfaces and arrays, are assigned, so what an interface
// holds stays shared.
//
// It writes into dst in place and copies slices of assigned kinds, such as
// []byte, with one reflect.Copy, so it allocates only the pointers, slices
// and maps of the copy, not a temporary per field, element or byte.
func deepCopy(src, dst reflect.Value) {
	switch src.Kind() {
	case reflect.Struct:
		// Assign every field, then replace the ones that need a deep copy.
		dst.Set(src)
		for i := range src.NumField() {
			// dst is settable, so its field is settable exactly when the
			// field is exported.
			field := dst.Field(i)
			if !field.CanSet() {
				t := src.Type()
				panic(fmt.Sprintf("deepCopy: unexported field %q in type %q", t.Field(i).Name, t.Name()))
			}
			if !assignedKind(field.Kind()) {
				deepCopy(src.Field(i), field)
			}
		}
	case reflect.Slice:
		if src.IsNil() {
			return
		}
		s := reflect.MakeSlice(src.Type(), src.Len(), src.Cap())
		if assignedKind(src.Type().Elem().Kind()) {
			reflect.Copy(s, src)
		} else {
			for i := range src.Len() {
				deepCopy(src.Index(i), s.Index(i))
			}
		}
		dst.Set(s)
	case reflect.Map:
		if src.IsNil() {
			return
		}
		t := src.Type()
		m := reflect.MakeMapWithSize(t, src.Len())
		// Copies of the key and value of each entry; SetMapIndex stores
		// its own copies of them, so they are reused.
		key, val := reflect.New(t.Key()).Elem(), reflect.New(t.Elem()).Elem()
		deepKey, deepVal := !assignedKind(t.Key().Kind()), !assignedKind(t.Elem().Kind())
		var iter reflect.MapIter
		iter.Reset(src)
		for iter.Next() {
			if deepKey {
				key.SetZero()
				deepCopy(iter.Key(), key)
			} else {
				key.SetIterKey(&iter)
			}
			if deepVal {
				val.SetZero()
				deepCopy(iter.Value(), val)
			} else {
				val.SetIterValue(&iter)
			}
			m.SetMapIndex(key, val)
		}
		dst.Set(m)
	case reflect.Pointer:
		if src.IsNil() {
			return
		}
		// Create a new pointer and deep copy the underlying value
		newPtr := reflect.New(src.Elem().Type())
		deepCopy(src.Elem(), newPtr.Elem())
		dst.Set(newPtr)
	default:
		// For basic types, direct assignment is sufficient
		dst.Set(src)
	}
}

// assignedKind reports whether deepCopy copies values of kind k by plain
// assignment.
func assignedKind(k reflect.Kind) bool {
	switch k {
	case reflect.Struct, reflect.Slice, reflect.Map, reflect.Pointer:
		return false
	}
	return true
}
