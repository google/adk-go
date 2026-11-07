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

package toolauth

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
	"sync"
	"unicode"
)

// The credential types follow adk-python's pydantic models, which derive from
// BaseModelWithConfig in auth_credential.py. Such a model writes each field
// under its camelCase alias, reads it under the alias or under the snake_case
// Python field name, and keeps the keys that match no field. decodeModel and
// encodeModel give a struct with camelCase json tags the same behavior.

// modelFields lists the names a model struct's fields are read under.
type modelFields struct {
	// aliases holds each field's json tag name, its camelCase alias.
	aliases map[string]bool
	// names maps a Python field name to its alias, where the two differ.
	names map[string]string
}

// modelFieldsCache maps a struct's reflect.Type to its *modelFields.
var modelFieldsCache sync.Map

func fieldsOf(t reflect.Type) *modelFields {
	if f, ok := modelFieldsCache.Load(t); ok {
		return f.(*modelFields)
	}
	f := &modelFields{aliases: map[string]bool{}, names: map[string]string{}}
	for i := range t.NumField() {
		alias, _, _ := strings.Cut(t.Field(i).Tag.Get("json"), ",")
		if alias == "" || alias == "-" {
			continue
		}
		f.aliases[alias] = true
		if name := pythonName(alias); name != alias {
			f.names[name] = alias
		}
	}
	modelFieldsCache.Store(t, f)
	return f
}

// pythonName returns the snake_case Python field name that pydantic's to_camel
// turns into alias: client_id for clientId, and auth_provider_x509_cert_url for
// authProviderX509CertUrl. Python calls a field type_ so as not to shadow the
// builtin, so the name for type is type_.
func pythonName(alias string) string {
	if alias == "type" {
		return "type_"
	}
	var b strings.Builder
	for _, r := range alias {
		if unicode.IsUpper(r) {
			b.WriteByte('_')
			r = unicode.ToLower(r)
		}
		b.WriteRune(r)
	}
	return b.String()
}

// decodeModel decodes the JSON object data into v, a pointer to a struct with
// camelCase json tags, and sets *extra to the keys that no field claims, or to
// nil when there are none. A field is read under its alias or its Python name,
// and the alias wins when the object has both. Each name in required must be
// present and not null, as a pydantic field without a default must be. Numbers
// in *extra are [json.Number], so they are written back exactly as read.
//
// Errors name the type and the field but never quote a value, which can be a
// secret.
func decodeModel(data []byte, v any, extra *map[string]any, typeName string, required ...string) error {
	return decode(data, v, extra, typeName, true, required)
}

// decodeOpenAPIModel decodes like decodeModel but reads each field under its
// json tag name only. The fastapi OpenAPI models that adk-python's auth
// schemes are built from accept no other spelling, so a snake_case key there
// is an unknown key and goes to *extra.
func decodeOpenAPIModel(data []byte, v any, extra *map[string]any, typeName string, required ...string) error {
	return decode(data, v, extra, typeName, false, required)
}

// decode implements decodeModel, and decodeOpenAPIModel when pythonNames is
// false.
func decode(data []byte, v any, extra *map[string]any, typeName string, pythonNames bool, required []string) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return decodeError(typeName, err)
	}
	fields := fieldsOf(reflect.TypeOf(v).Elem())
	known := make(map[string]json.RawMessage, len(raw))
	var unknown map[string]any
	for key, val := range raw {
		alias := key
		if !fields.aliases[key] {
			name, isName := fields.names[key]
			if !isName || !pythonNames {
				dec := json.NewDecoder(bytes.NewReader(val))
				dec.UseNumber()
				var x any
				if err := dec.Decode(&x); err != nil {
					return decodeError(typeName, err)
				}
				if unknown == nil {
					unknown = map[string]any{}
				}
				unknown[key] = x
				continue
			}
			if _, both := raw[name]; both {
				continue
			}
			alias = name
		}
		known[alias] = val
	}
	for _, name := range required {
		if val, ok := known[name]; !ok || bytes.Equal(bytes.TrimSpace(val), []byte("null")) {
			return fmt.Errorf("toolauth: %s: missing required field %s", typeName, name)
		}
	}
	b, err := json.Marshal(known)
	if err != nil {
		return decodeError(typeName, err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		return decodeError(typeName, err)
	}
	*extra = unknown
	return nil
}

// decodeError reports err, from decoding a typeName, without the value that
// caused it: encoding/json quotes a number in full when it overflows an
// integer field, such as 1e100 for expiresAt. An error that encoding/json did
// not make came from a nested model's UnmarshalJSON, which has already left
// the value out.
func decodeError(typeName string, err error) error {
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &typeErr) {
		got, _, _ := strings.Cut(typeErr.Value, " ")
		if typeErr.Field == "" {
			return fmt.Errorf("toolauth: %s: want a JSON object, got %s", typeName, got)
		}
		return fmt.Errorf("toolauth: %s: field %s: cannot use a JSON %s as %s", typeName, typeErr.Field, got, typeErr.Type)
	}
	var syntaxErr *json.SyntaxError
	if errors.As(err, &syntaxErr) {
		return fmt.Errorf("toolauth: %s: malformed JSON at offset %d", typeName, syntaxErr.Offset)
	}
	return err
}

// encodeModel writes v, a struct with camelCase json tags, followed by the
// keys of extra in sorted order. A key of extra that is one of v's alias or
// Python field names is skipped, so that no field is written twice.
func encodeModel(v any, extra map[string]any) ([]byte, error) {
	return encode(v, extra, true)
}

// encodeOpenAPIModel writes like encodeModel, but skips only the keys of extra
// that are one of v's json tag names, since an OpenAPI model has no other
// names.
func encodeOpenAPIModel(v any, extra map[string]any) ([]byte, error) {
	return encode(v, extra, false)
}

// encode implements encodeModel, and encodeOpenAPIModel when pythonNames is
// false.
func encode(v any, extra map[string]any, pythonNames bool) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil || len(extra) == 0 {
		return b, err
	}
	fields := fieldsOf(reflect.TypeOf(v))
	var buf bytes.Buffer
	buf.Write(b[:len(b)-1]) // everything but the closing brace
	wrote := len(b) > len("{}")
	for _, key := range slices.Sorted(maps.Keys(extra)) {
		if _, isName := fields.names[key]; (isName && pythonNames) || fields.aliases[key] {
			continue
		}
		k, err := json.Marshal(key)
		if err != nil {
			return nil, err
		}
		val, err := json.Marshal(extra[key])
		if err != nil {
			return nil, fmt.Errorf("toolauth: encode an Extra value: %w", err)
		}
		if wrote {
			buf.WriteByte(',')
		}
		buf.Write(k)
		buf.WriteByte(':')
		buf.Write(val)
		wrote = true
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}
