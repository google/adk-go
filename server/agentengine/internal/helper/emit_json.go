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

package helper

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// flush flushes buffered data to the client.
func flush(rw http.ResponseWriter) error {
	w := rw
	for {
		switch t := w.(type) {
		case interface{ FlushError() error }:
			return t.FlushError()
		case http.Flusher:
			t.Flush()
			return nil
		case rwUnwrapper:
			w = t.Unwrap()
		default:
			return fmt.Errorf("not supported type to flush: %v %T", t, t)
		}
	}
}

// this interface is used to get to the underlying http.ResponseWriter if wrapped.
type rwUnwrapper interface {
	Unwrap() http.ResponseWriter
}

// EmitJSON emits a line with JSON for an event.
func EmitJSON(rw http.ResponseWriter, o any) error {
	snake := ConvertSnake(o)
	err := json.NewEncoder(rw).Encode(snake)
	if err != nil {
		return fmt.Errorf("failed to encode SSE response chunk: %w", err)
	}
	err = flush(rw)
	if err != nil {
		return fmt.Errorf("failed to flush: %w", err)
	}

	return nil
}

// EmitJSONError emits a line with json describing the error
func EmitJSONError(rw http.ResponseWriter, origError error) error {
	jsonErr := map[string]any{
		"error": origError.Error(),
	}
	err := EmitJSON(rw, jsonErr)
	if err != nil {
		return fmt.Errorf("failed to emit error: %w", err)
	}
	return nil
}
