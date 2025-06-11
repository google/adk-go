// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonschema_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/adk-go/internal/jsonschema"
)

func forType[T any]() *jsonschema.Schema {
	s, err := jsonschema.For[T]()
	if err != nil {
		panic(err)
	}
	return s
}

func TestForType(t *testing.T) {
	type schema = jsonschema.Schema
	tests := []struct {
		name string
		got  *jsonschema.Schema
		want *jsonschema.Schema
	}{
		{"string", forType[string](), &schema{Type: "string"}},
		{"int", forType[int](), &schema{Type: "integer"}},
		{"int16", forType[int16](), &schema{Type: "integer"}},
		{"uint32", forType[int16](), &schema{Type: "integer"}},
		{"float64", forType[float64](), &schema{Type: "number"}},
		{"bool", forType[bool](), &schema{Type: "boolean"}},
		{"intmap", forType[map[string]int](), &schema{
			Type:                 "object",
			AdditionalProperties: &schema{Type: "integer"},
		}},
		{"anymap", forType[map[string]any](), &schema{
			Type:                 "object",
			AdditionalProperties: &schema{},
		}},
		{
			"struct",
			forType[struct {
				F           int `json:"f"`
				G           []float64
				P           *bool
				Skip        string `json:"-"`
				NoSkip      string `json:",omitempty"`
				unexported  float64
				unexported2 int `json:"No"`
			}](),
			&schema{
				Type: "object",
				Properties: map[string]*schema{
					"f":      {Type: "integer"},
					"G":      {Type: "array", Items: &schema{Type: "number"}},
					"P":      {Types: []string{"null", "boolean"}},
					"NoSkip": {Type: "string"},
				},
				Required:             []string{"f", "G", "P"},
				AdditionalProperties: &jsonschema.Schema{Not: &jsonschema.Schema{}},
			},
		},
		{
			"no sharing",
			forType[struct{ X, Y int }](),
			&schema{
				Type: "object",
				Properties: map[string]*schema{
					"X": {Type: "integer"},
					"Y": {Type: "integer"},
				},
				Required:             []string{"X", "Y"},
				AdditionalProperties: &jsonschema.Schema{Not: &jsonschema.Schema{}},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if diff := cmp.Diff(test.want, test.got, cmpopts.IgnoreUnexported(jsonschema.Schema{})); diff != "" {
				t.Fatalf("ForType mismatch (-want +got):\n%s", diff)
			}
			// These schemas should all resolve.
			if _, err := test.got.Resolve(nil); err != nil {
				t.Fatalf("Resolving: %v", err)
			}
		})
	}
}
