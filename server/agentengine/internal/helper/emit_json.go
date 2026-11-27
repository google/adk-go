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
	"errors"
	"fmt"
	"net/http"
	"reflect"

	"google.golang.org/adk/v2/session"
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

// EventForStream prepares the optional workflow output independently of the
// event envelope. If it cannot be represented as JSON, the event is sent
// without output. The persisted event is left untouched.
func EventForStream(event *session.Event) *session.Event {
	if event.Output == nil {
		return event
	}
	wireEvent := *event
	wireEvent.Output = nil
	output, err := convertSnake(".output", event.Output)
	if err != nil {
		if !canFallback(err, event.Output) {
			return &wireEvent
		}
		output = event.Output
	}
	if output != nil && notEmpty(output) {
		if data, err := json.Marshal(output); err == nil {
			wireEvent.Output = json.RawMessage(data)
		}
	}
	return &wireEvent
}

func canFallback(err error, input any) bool {
	return !errors.Is(err, errSnakeCycle) && !errors.Is(err, errSnakeDepth) &&
		withinJSONDepth(reflect.ValueOf(input), maxSnakeDepth)
}

// A failed conversion may have stopped before visiting another field. Check
// the whole value before handing it to encoding/json.
func withinJSONDepth(v reflect.Value, remaining int) bool {
	if !v.IsValid() {
		return true
	}
	if remaining == 0 {
		return false
	}
	remaining--
	switch v.Kind() {
	case reflect.Interface, reflect.Pointer:
		return withinJSONDepth(v.Elem(), remaining)
	case reflect.Array, reflect.Slice:
		for i := range v.Len() {
			if !withinJSONDepth(v.Index(i), remaining) {
				return false
			}
		}
	case reflect.Map:
		entries := v.MapRange()
		for entries.Next() {
			if !withinJSONDepth(entries.Value(), remaining) {
				return false
			}
		}
	case reflect.Struct:
		for i := range v.NumField() {
			field := v.Type().Field(i)
			fv := v.Field(i)
			if field.Tag.Get("json") == "-" ||
				(!field.IsExported() && (!field.Anonymous || reflect.Indirect(fv).Kind() != reflect.Struct)) {
				continue
			}
			if !withinJSONDepth(fv, remaining) {
				return false
			}
		}
	}
	return true
}

// ErrEventEncoding means no bytes were written for an unencodable event.
var ErrEventEncoding = errors.New("failed to encode Agent Engine event")

// EmitJSON emits a line with JSON for an event.
func EmitJSON(rw http.ResponseWriter, o any) error {
	snake, err := convertSnake("", o)
	if err != nil {
		if !canFallback(err, o) {
			return fmt.Errorf("%w: %w", ErrEventEncoding, err)
		}
		snake = o
	}
	data, err := json.Marshal(snake)
	if err != nil {
		return ErrEventEncoding
	}
	if _, err := rw.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("failed to write SSE response chunk: %w", err)
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
