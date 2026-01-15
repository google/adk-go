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

package utils

import (
	"reflect"
	"testing"

	"google.golang.org/genai"
)

func TestMatchType(t *testing.T) {
	tests := []struct {
		name      string
		value     any
		schema    *genai.Schema
		isInput   bool
		wantMatch bool
		wantErr   bool
	}{
		{
			name:      "nil schema",
			value:     "test",
			schema:    nil,
			isInput:   true,
			wantMatch: false,
			wantErr:   true,
		},
		{
			name:      "nil value",
			value:     nil,
			schema:    &genai.Schema{Type: genai.TypeString},
			isInput:   true,
			wantMatch: false,
			wantErr:   false,
		},
		{
			name:      "string match",
			value:     "test",
			schema:    &genai.Schema{Type: genai.TypeString},
			isInput:   true,
			wantMatch: true,
			wantErr:   false,
		},
		{
			name:      "string mismatch",
			value:     123.0,
			schema:    &genai.Schema{Type: genai.TypeString},
			isInput:   true,
			wantMatch: false,
			wantErr:   false,
		},
		{
			name:      "integer match",
			value:     123.0,
			schema:    &genai.Schema{Type: genai.TypeInteger},
			isInput:   true,
			wantMatch: true,
			wantErr:   false,
		},
		{
			name:      "integer mismatch float",
			value:     123.45,
			schema:    &genai.Schema{Type: genai.TypeInteger},
			isInput:   true,
			wantMatch: false,
			wantErr:   false,
		},
		{
			name:      "integer mismatch type",
			value:     "123",
			schema:    &genai.Schema{Type: genai.TypeInteger},
			isInput:   true,
			wantMatch: false,
			wantErr:   false,
		},
		{
			name:      "number match",
			value:     123.45,
			schema:    &genai.Schema{Type: genai.TypeNumber},
			isInput:   true,
			wantMatch: true,
			wantErr:   false,
		},
		{
			name:      "number mismatch",
			value:     "123.45",
			schema:    &genai.Schema{Type: genai.TypeNumber},
			isInput:   true,
			wantMatch: false,
			wantErr:   false,
		},
		{
			name:      "boolean match",
			value:     true,
			schema:    &genai.Schema{Type: genai.TypeBoolean},
			isInput:   true,
			wantMatch: true,
			wantErr:   false,
		},
		{
			name:      "boolean mismatch",
			value:     "true",
			schema:    &genai.Schema{Type: genai.TypeBoolean},
			isInput:   true,
			wantMatch: false,
			wantErr:   false,
		},
		{
			name:      "array match",
			value:     []any{"a", "b"},
			schema:    &genai.Schema{Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
			isInput:   true,
			wantMatch: true,
			wantErr:   false,
		},
		{
			name:      "array mismatch type",
			value:     "not an array",
			schema:    &genai.Schema{Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
			isInput:   true,
			wantMatch: false,
			wantErr:   false,
		},
		{
			name:      "array mismatch item type",
			value:     []any{"a", 1.0},
			schema:    &genai.Schema{Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
			isInput:   true,
			wantMatch: false,
			wantErr:   false,
		},
		{
			name:      "array missing items",
			value:     []any{"a", "b"},
			schema:    &genai.Schema{Type: genai.TypeArray},
			isInput:   true,
			wantMatch: false,
			wantErr:   true,
		},
		{
			name:      "object match",
			value:     map[string]any{"foo": "bar"},
			schema:    &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{"foo": {Type: genai.TypeString}}},
			isInput:   true,
			wantMatch: true,
			wantErr:   false,
		},
		{
			name:      "object mismatch type",
			value:     "not an object",
			schema:    &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{"foo": {Type: genai.TypeString}}},
			isInput:   true,
			wantMatch: false,
			wantErr:   false,
		},
		{
			name:      "object mismatch property type",
			value:     map[string]any{"foo": 123.0},
			schema:    &genai.Schema{Type: genai.TypeObject, Properties: map[string]*genai.Schema{"foo": {Type: genai.TypeString}}},
			isInput:   true,
			wantMatch: false,
			wantErr:   true, //This will fail ValidateMapOnSchema, which returns error
		},
		{
			name:      "unsupported type",
			value:     123,
			schema:    &genai.Schema{Type: "UNSUPPORTED"},
			isInput:   true,
			wantMatch: false,
			wantErr:   true,
		},
		{
			name:      "lowercase type in schema",
			value:     "test",
			schema:    &genai.Schema{Type: "string"},
			isInput:   true,
			wantMatch: true,
			wantErr:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotMatch, err := matchType(tt.value, tt.schema, tt.isInput)
			if (err != nil) != tt.wantErr {
				t.Errorf("matchType() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if gotMatch != tt.wantMatch {
				t.Errorf("matchType() = %v, want %v", gotMatch, tt.wantMatch)
			}
		})
	}
}

func TestValidateMapOnSchema(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"str_field": {Type: genai.TypeString},
			"int_field": {Type: genai.TypeInteger},
		},
		Required: []string{"str_field"},
	}
	schemaNilProps := &genai.Schema{
		Type: genai.TypeObject,
	}

	tests := []struct {
		name    string
		args    map[string]any
		schema  *genai.Schema
		isInput bool
		wantErr bool
	}{
		{
			name:    "valid map",
			args:    map[string]any{"str_field": "hello", "int_field": 123.0},
			schema:  schema,
			isInput: true,
			wantErr: false,
		},
		{
			name:    "valid map with only required fields",
			args:    map[string]any{"str_field": "hello"},
			schema:  schema,
			isInput: true,
			wantErr: false,
		},
		{
			name:    "missing required field",
			args:    map[string]any{"int_field": 123.0},
			schema:  schema,
			isInput: true,
			wantErr: true,
		},
		{
			name:    "extra field",
			args:    map[string]any{"str_field": "hello", "extra": "field"},
			schema:  schema,
			isInput: true,
			wantErr: true,
		},
		{
			name:    "type mismatch",
			args:    map[string]any{"str_field": 123.0},
			schema:  schema,
			isInput: true,
			wantErr: true,
		},
		{
			name:    "nil schema",
			args:    map[string]any{"str_field": "hello"},
			schema:  nil,
			isInput: true,
			wantErr: true,
		},
		{
			name:    "nil properties and no args",
			args:    map[string]any{},
			schema:  schemaNilProps,
			isInput: true,
			wantErr: false,
		},
		{
			name:    "nil properties and some args",
			args:    map[string]any{"some": "arg"},
			schema:  schemaNilProps,
			isInput: true,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateMapOnSchema(tt.args, tt.schema, tt.isInput); (err != nil) != tt.wantErr {
				t.Errorf("ValidateMapOnSchema() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateOutputSchema(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"result": {Type: genai.TypeString},
		},
		Required: []string{"result"},
	}

	tests := []struct {
		name       string
		output     string
		schema     *genai.Schema
		wantOutput map[string]any
		wantErr    bool
	}{
		{
			name:       "valid output",
			output:     `{"result": "success"}`,
			schema:     schema,
			wantOutput: map[string]any{"result": "success"},
			wantErr:    false,
		},
		{
			name:       "invalid json",
			output:     `{"result": "success"`,
			schema:     schema,
			wantOutput: nil,
			wantErr:    true,
		},
		{
			name:       "schema mismatch",
			output:     `{"wrong_key": "failure"}`,
			schema:     schema,
			wantOutput: nil,
			wantErr:    true,
		},
		{
			name:       "nil schema",
			output:     `{"result": "success"}`,
			schema:     nil,
			wantOutput: nil,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotOutput, err := ValidateOutputSchema(tt.output, tt.schema)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateOutputSchema() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && !reflect.DeepEqual(gotOutput, tt.wantOutput) {
				t.Errorf("ValidateOutputSchema() = %v, want %v", gotOutput, tt.wantOutput)
			}
		})
	}
}
