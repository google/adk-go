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

package telemetry

import (
	"fmt"

	"go.opentelemetry.io/otel/log"
)

// toLogValue converts a JSON value to a log.Value.
// From [encoding/json.Unmarshal] documentation:
// To unmarshal JSON into an interface value,
// Unmarshal stores one of these in the interface value:
//
//   - bool, for JSON booleans
//   - float64, for JSON numbers
//   - string, for JSON strings
//   - []any, for JSON arrays
//   - map[string]any, for JSON objects
//   - nil for JSON null
func toLogValue(v any) log.Value {
	switch val := v.(type) {
	case nil:
		return log.Value{}
	case string:
		return log.StringValue(val)
	case bool:
		return log.BoolValue(val)
	case float64:
		return log.Float64Value(val)
	case int:
		return log.IntValue(val)
	case []any:
		values := make([]log.Value, 0, len(val))
		for _, item := range val {
			values = append(values, toLogValue(item))
		}
		return log.SliceValue(values...)
	case map[string]any:
		kvs := make([]log.KeyValue, 0, len(val))
		for k, v := range val {
			kvs = append(kvs, log.KeyValue{Key: k, Value: toLogValue(v)})
		}
		return log.MapValue(kvs...)
	default:
		// Fallback for other types
		return log.StringValue(fmt.Sprintf("%v", val))
	}
}

// FromLogValue converts a log.Value to golang type. See [toLogValue] for more details.
func FromLogValue(v log.Value) any {
	switch v.Kind() {
	case log.KindString:
		return v.AsString()
	case log.KindInt64:
		return v.AsInt64()
	case log.KindFloat64:
		return v.AsFloat64()
	case log.KindBool:
		return v.AsBool()
	case log.KindBytes:
		return v.AsBytes()
	case log.KindMap:
		m := make(map[string]any)
		for _, kv := range v.AsMap() {
			m[kv.Key] = FromLogValue(kv.Value)
		}
		return m
	case log.KindSlice:
		s := make([]any, 0)
		for _, v := range v.AsSlice() {
			s = append(s, FromLogValue(v))
		}
		return s
	case log.KindEmpty:
		return nil
	default:
		// Try to handle this as gracefully as possible.
		//
		// Don't panic here. The goal here is to have developers find this
		// first if a slog.Kind is is not handled. It is
		// preferable to have user's open issue asking why their attributes
		// have a "unhandled: " prefix than say that their code is panicking.
		return fmt.Sprintf("<unhandled log.Kind: %s>", v.Kind())
	}
}
