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
	"reflect"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

// basicRequestProcessor populates the LLMRequest
// with the agent's LLM generation configs.
func basicRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest) error {
	// reference: adk-python src/google/adk/flows/llm_flows/basic.py

	llmAgent := asLLMAgent(ctx.Agent())
	if llmAgent == nil {
		return nil // do nothing.
	}
	req.Config = clone(llmAgent.internal().GenerateContentConfig)
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if llmAgent.internal().OutputSchema != nil {
		req.Config.ResponseSchema = llmAgent.internal().OutputSchema
		req.Config.ResponseMIMEType = "application/json"
	}
	// TODO: missing features
	//  populate LLMRequest LiveConnectConfig setting
	return nil
}

// clone returns a deep copy of the src.
// NOTE: this does not work for types with unexported fields.
func clone[M any](src M) M {
	val := reflect.ValueOf(src)

	// Handle nil pointers
	if val.Kind() == reflect.Ptr && val.IsNil() {
		var zero M
		return zero
	}

	srcIsPointer := val.Kind() == reflect.Ptr

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

// deepCopy copies src to dst using reflect.
func deepCopy(src, dst reflect.Value) {
	switch src.Kind() {
	case reflect.Struct:
		t := src.Type()
		for i := 0; i < src.NumField(); i++ {
			if !t.Field(i).IsExported() {
				panic(fmt.Sprintf("deepCopy: unexported field %q in type %q", t.Field(i).Name, t.Name()))
			}
			// Create a copy of the field and set it on the destination struct
			fieldCopy := reflect.New(src.Field(i).Type()).Elem()
			deepCopy(src.Field(i), fieldCopy)
			dst.Field(i).Set(fieldCopy)
		}
	case reflect.Slice:
		if src.IsNil() {
			return
		}
		dst.Set(reflect.MakeSlice(src.Type(), src.Len(), src.Cap()))
		for i := 0; i < src.Len(); i++ {
			// Create a copy of each element and set it in the new slice
			elemCopy := reflect.New(src.Index(i).Type()).Elem()
			deepCopy(src.Index(i), elemCopy)
			dst.Index(i).Set(elemCopy)
		}
	case reflect.Map:
		if src.IsNil() {
			return
		}
		dst.Set(reflect.MakeMap(src.Type()))
		for _, key := range src.MapKeys() {
			// Create copies of the key and value and set them in the new map
			keyCopy := reflect.New(key.Type()).Elem()
			deepCopy(key, keyCopy)
			valCopy := reflect.New(src.MapIndex(key).Type()).Elem()
			deepCopy(src.MapIndex(key), valCopy)
			dst.SetMapIndex(keyCopy, valCopy)
		}
	case reflect.Ptr:
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
