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

// Package schemacache remembers the JSON encoding of tool schemas that can no
// longer change, so a model adapter can send them without encoding them again
// on every call. jsonschema.Schema's MarshalJSON is slow enough that encoding
// the schemas of three small tools costs more than the rest of a Gemini call
// with a short history.
package schemacache

import (
	"encoding/json"
	"runtime"
	"slices"
	"sync"
	"weak"

	"github.com/google/jsonschema-go/jsonschema"
)

// encoded maps a remembered schema to its encoding. The keys are weak, and a
// cleanup deletes an entry once its schema is collected, so tools built per
// request do not grow the map.
var encoded sync.Map // weak.Pointer[jsonschema.Schema] -> json.RawMessage

// Remember records the encoding of s, replacing any earlier one. s must not
// change afterwards, which jsonschema already requires of a schema once it has
// been resolved. A schema changed anyway is still sent with the encoding
// recorded here, until it is remembered again.
func Remember(s *jsonschema.Schema) {
	if s == nil {
		return
	}
	b, err := json.Marshal(s)
	if err != nil {
		// Not remembered, so the adapter encodes s itself and reports the
		// error as it would have.
		return
	}
	key := weak.Make(s)
	if _, loaded := encoded.Swap(key, json.RawMessage(b)); !loaded {
		runtime.AddCleanup(s, func(k weak.Pointer[jsonschema.Schema]) { encoded.Delete(k) }, key)
	}
}

// Encoded returns the remembered encoding of v, and false when v is not a
// remembered schema.
func Encoded(v any) (json.RawMessage, bool) {
	s, ok := v.(*jsonschema.Schema)
	if !ok {
		return nil, false
	}
	// A nil s makes the zero weak pointer, which Remember never stores.
	raw, ok := encoded.Load(weak.Make(s))
	if !ok {
		return nil, false
	}
	b := raw.(json.RawMessage)
	// Capped so an append by a caller cannot write into the shared array.
	return slices.Clip(b), true
}
