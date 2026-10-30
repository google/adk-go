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
	"reflect"
	"time"
)

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

// cloneJSONError hides customer content while preserving the underlying error.
type cloneJSONError struct{ err error }

func (e *cloneJSONError) Error() string {
	return "clone JSON value: serialization failed"
}

func (e *cloneJSONError) Unwrap() error { return e.err }

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
			if err := deepCopy(src.Field(i), dst.Field(i), depth+1); err != nil {
				return err
			}
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
				return &cloneJSONError{err: err}
			}
			valueCopy = reflect.ValueOf(json.RawMessage(raw))
		}
		dst.Set(valueCopy)
	case reflect.Array, reflect.Slice:
		if src.Kind() == reflect.Slice {
			if src.IsNil() {
				return nil
			}
			dst.Set(reflect.MakeSlice(src.Type(), src.Len(), src.Cap()))
		}
		for i := 0; i < src.Len(); i++ {
			if err := deepCopy(src.Index(i), dst.Index(i), depth+1); err != nil {
				return err
			}
		}
	case reflect.Map:
		if src.IsNil() {
			return nil
		}
		dst.Set(reflect.MakeMapWithSize(src.Type(), src.Len()))
		for it := src.MapRange(); it.Next(); {
			key, value := it.Key(), it.Value()
			keyCopy := reflect.New(key.Type()).Elem()
			if err := deepCopy(key, keyCopy, depth+1); err != nil {
				return err
			}
			if !keyCopy.Comparable() {
				return fmt.Errorf("cloned map key of type %v is not comparable", key.Type())
			}
			valCopy := reflect.New(value.Type()).Elem()
			if err := deepCopy(value, valCopy, depth+1); err != nil {
				return err
			}
			dst.SetMapIndex(keyCopy, valCopy)
		}
	case reflect.Pointer:
		if src.IsNil() {
			return nil
		}
		newPtr := reflect.New(src.Elem().Type())
		if err := deepCopy(src.Elem(), newPtr.Elem(), depth+1); err != nil {
			return err
		}
		dst.Set(newPtr)
	default:
		dst.Set(src)
	}
	return nil
}
