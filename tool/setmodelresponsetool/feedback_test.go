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

package setmodelresponsetool_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/toolinternal"
)

const (
	feedbackHead = "Validation Error found:\n"
	feedbackTail = "Recall the set_model_response function correctly, fix the errors, and call it again with all required fields using the correct types."
)

// checkFeedback runs tl with args and checks that it fails with exactly
// wantErr and stores nothing.
func checkFeedback(t *testing.T, tl toolinternal.FunctionTool, args any, wantErr string) {
	t.Helper()
	ctx := newToolContext(t)

	got, err := tl.Run(ctx, args)

	if err == nil {
		t.Fatalf("Run(%#v) = %v, nil; want error", args, got)
	}
	if err.Error() != wantErr {
		t.Errorf("Run(%#v) error:\n%s\nwant:\n%s", args, err.Error(), wantErr)
	}
	if got != nil {
		t.Errorf("Run(%#v) result = %v, want nil", args, got)
	}
	if ctx.Actions().SetModelResponse != nil {
		t.Errorf("Run(%#v) stored %#v, want nil", args, ctx.Actions().SetModelResponse)
	}
}

func TestRun_ValidationFeedback_AllIssueKinds(t *testing.T) {
	fields := newTool(t, &genai.Schema{
		Type:  genai.TypeObject,
		Title: "Fields",
		Properties: map[string]*genai.Schema{
			"s":    {Type: genai.TypeString},
			"n":    {Type: genai.TypeInteger},
			"f":    {Type: genai.TypeNumber},
			"b":    {Type: genai.TypeBoolean},
			"tags": {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
			"addr": {
				Type:       genai.TypeObject,
				Title:      "Address",
				Properties: map[string]*genai.Schema{"zip": {Type: genai.TypeString}},
			},
			"addr0": {
				Type:       genai.TypeObject,
				Properties: map[string]*genai.Schema{"zip": {Type: genai.TypeString}},
			},
			"meta": {Type: genai.TypeObject},
			"e1":   {Type: genai.TypeString, Enum: []string{"a"}},
			"e2":   {Type: genai.TypeString, Enum: []string{"a", "b"}},
			"e3":   {Type: genai.TypeString, Enum: []string{"a", "b", "c"}},
		},
	})

	singleField := []struct {
		name   string
		field  string
		inputs []any
		want   string
	}{
		{"string_type", "s", []any{123, 1.5, true, nil}, "s\n  Input should be a valid string [type=string_type]\n"},
		{"int_from_float", "n", []any{float64(25.5), float32(1.5)}, "n\n  Input should be a valid integer, got a number with a fractional part [type=int_from_float]\n"},
		{
			"int_parsing", "n",
			[]any{"abc", "2.5", ".0", "0.", "1e2", "", "   ", "25.", "_1", "1__0", "1.0_0"},
			"n\n  Input should be a valid integer, unable to parse string as an integer [type=int_parsing]\n",
		},
		{"int_parsing_size", "n", []any{float64(1e20), float64(-9223372036854775808), strings.Repeat("9", 400)}, "n\n  Unable to parse input string as an integer, exceeded maximum size [type=int_parsing_size]\n"},
		{"int_finite_number", "n", []any{math.NaN(), math.Inf(1)}, "n\n  Input should be a finite number [type=finite_number]\n"},
		{"int_type", "n", []any{nil, []any{1}, map[string]any{}}, "n\n  Input should be a valid integer [type=int_type]\n"},
		{
			"float_parsing", "f",
			[]any{"abc", "", "   ", "nan", "NaN", "inf", "-inf", "+inf", "infinity", "1e400", "1_0", "1.5e"},
			"f\n  Input should be a valid number, unable to parse string as a number [type=float_parsing]\n",
		},
		{"float_finite_number", "f", []any{math.NaN(), math.Inf(1)}, "f\n  Input should be a finite number [type=finite_number]\n"},
		{"float_type", "f", []any{nil, []any{}, map[string]any{}}, "f\n  Input should be a valid number [type=float_type]\n"},
		{
			"bool_parsing", "b",
			[]any{int(2), int(-1), float64(2.0), uint(7), "maybe", " true", "true ", "1.0", ""},
			"b\n  Input should be a valid boolean, unable to interpret input [type=bool_parsing]\n",
		},
		{
			"bool_type", "b",
			[]any{float64(0.5), float64(1e20), math.NaN(), uint64(math.MaxUint64), nil, []any{}, map[string]any{}},
			"b\n  Input should be a valid boolean [type=bool_type]\n",
		},
		{"list_type", "tags", []any{"not_a_list", 123, nil, map[string]any{"a": 1}}, "tags\n  Input should be a valid list [type=list_type]\n"},
		{"model_type_with_title", "addr", []any{"not_a_map"}, "addr\n  Input should be a valid dictionary or instance of Address [type=model_type]\n"},
		{"model_type_default_title", "addr0", []any{123}, "addr0\n  Input should be a valid dictionary or instance of object [type=model_type]\n"},
		{"dict_type", "meta", []any{"not_a_map", []any{1}}, "meta\n  Input should be a valid dictionary [type=dict_type]\n"},
		{"enum_1_member", "e1", []any{"z"}, "e1\n  Input should be 'a' [type=enum]\n"},
		{"enum_2_members", "e2", []any{"z"}, "e2\n  Input should be 'a' or 'b' [type=enum]\n"},
		{"enum_3_members", "e3", []any{123}, "e3\n  Input should be 'a', 'b' or 'c' [type=enum]\n"},
	}
	for _, tc := range singleField {
		t.Run(tc.name, func(t *testing.T) {
			want := feedbackHead + "1 validation error for Fields\n" + tc.want + feedbackTail
			for _, in := range tc.inputs {
				t.Run(fmt.Sprintf("%#v", in), func(t *testing.T) {
					checkFeedback(t, fields, map[string]any{tc.field: in}, want)
				})
			}
		})
	}

	t.Run("missing_required_and_feedback_format", func(t *testing.T) {
		person := newTool(t, &genai.Schema{
			Type:  genai.TypeObject,
			Title: "PersonSchema",
			Properties: map[string]*genai.Schema{
				"name": {Type: genai.TypeString},
				"age":  {Type: genai.TypeInteger},
				"city": {Type: genai.TypeString},
			},
			Required: []string{"name", "age", "city"},
		})
		const want = "Validation Error found:\n" +
			"1 validation error for PersonSchema\n" +
			"age\n" +
			"  Field required [type=missing]\n" +
			"Recall the set_model_response function correctly, fix the errors, and call it again with all required fields using the correct types."

		checkFeedback(t, person, map[string]any{"name": "Charlie", "city": "Denver"}, want)
	})

	t.Run("multiple_issues_and_ordering", func(t *testing.T) {
		order := newTool(t, &genai.Schema{
			Type:             genai.TypeObject,
			Title:            "Order",
			PropertyOrdering: []string{"second", "first"},
			Properties: map[string]*genai.Schema{
				"first":  {Type: genai.TypeInteger},
				"second": {Type: genai.TypeString},
				"third":  {Type: genai.TypeString},
				"address": {
					Type:       genai.TypeObject,
					Title:      "Address",
					Properties: map[string]*genai.Schema{"zip": {Type: genai.TypeInteger}},
				},
				"tags": {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
			},
			Required: []string{"second"},
		})
		args := map[string]any{
			"first":   "bad_int",
			"third":   99,
			"address": map[string]any{"zip": "bad"},
			"tags":    []any{"ok", 42},
		}
		want := feedbackHead +
			"5 validation errors for Order\n" +
			"second\n  Field required [type=missing]\n" +
			"first\n  Input should be a valid integer, unable to parse string as an integer [type=int_parsing]\n" +
			"address.zip\n  Input should be a valid integer, unable to parse string as an integer [type=int_parsing]\n" +
			"tags.1\n  Input should be a valid string [type=string_type]\n" +
			"third\n  Input should be a valid string [type=string_type]\n" +
			feedbackTail

		checkFeedback(t, order, args, want)
	})

	items := newTool(t, &genai.Schema{
		Type: genai.TypeArray,
		Items: &genai.Schema{
			Type:  genai.TypeObject,
			Title: "ItemSchema",
			Properties: map[string]*genai.Schema{
				"id":   {Type: genai.TypeInteger},
				"name": {Type: genai.TypeString},
			},
			Required: []string{"id", "name"},
		},
	})

	t.Run("record_list_item_errors", func(t *testing.T) {
		want := feedbackHead +
			"2 validation errors for list[ItemSchema]\n" +
			"0\n  Input should be a valid dictionary or instance of ItemSchema [type=model_type]\n" +
			"1.id\n  Input should be a valid integer, unable to parse string as an integer [type=int_parsing]\n" +
			feedbackTail
		checkFeedback(t, items, map[string]any{"items": []any{"not_a_map", map[string]any{"id": "bad", "name": "ok"}}}, want)
	})

	t.Run("record_list_nil_item", func(t *testing.T) {
		want := feedbackHead +
			"1 validation error for list[ItemSchema]\n" +
			"0\n  Input should be a valid dictionary or instance of ItemSchema [type=model_type]\n" +
			feedbackTail
		checkFeedback(t, items, map[string]any{"items": []any{nil}}, want)
	})

	t.Run("missing_items", func(t *testing.T) {
		const want = "Validation Error found:\n" +
			"1 validation error for set_model_response\n" +
			"items\n" +
			"  Field required [type=missing]\n" +
			"Recall the set_model_response function correctly, fix the errors, and call it again with all required fields using the correct types."
		checkFeedback(t, items, map[string]any{}, want)
	})

	t.Run("non_list_items", func(t *testing.T) {
		want := feedbackHead +
			"1 validation error for list[ItemSchema]\n" +
			"  Input should be a valid list [type=list_type]\n" +
			feedbackTail
		for _, in := range []any{"bad", 5, map[string]any{"a": 1}, nil} {
			t.Run(fmt.Sprintf("%#v", in), func(t *testing.T) {
				checkFeedback(t, items, map[string]any{"items": in}, want)
			})
		}
	})

	t.Run("null_items_nullable_root", func(t *testing.T) {
		nullableItems := newTool(t, &genai.Schema{
			Type:     genai.TypeArray,
			Nullable: new(true),
			Items: &genai.Schema{
				Type:       genai.TypeObject,
				Title:      "ItemSchema",
				Properties: map[string]*genai.Schema{"id": {Type: genai.TypeInteger}},
				Required:   []string{"id"},
			},
		})
		want := feedbackHead +
			"1 validation error for list[ItemSchema]\n" +
			"  Input should be a valid list [type=list_type]\n" +
			feedbackTail
		checkFeedback(t, nullableItems, map[string]any{"items": nil}, want)
	})

	t.Run("missing_or_null_response", func(t *testing.T) {
		const want = "Validation Error found:\n" +
			"1 validation error for set_model_response\n" +
			"response\n" +
			"  Field required [type=missing]\n" +
			"Recall the set_model_response function correctly, fix the errors, and call it again with all required fields using the correct types."
		tools := map[string]toolinternal.FunctionTool{
			"value": newTool(t, &genai.Schema{Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}}),
			"raw":   newRawTool(t, &jsonschema.Schema{Type: "object"}),
		}
		for name, tl := range tools {
			t.Run(name+"_absent", func(t *testing.T) {
				checkFeedback(t, tl, map[string]any{}, want)
			})
			t.Run(name+"_null", func(t *testing.T) {
				checkFeedback(t, tl, map[string]any{"response": nil}, want)
			})
		}
	})

	t.Run("non_map_args", func(t *testing.T) {
		checkFeedback(t, fields, "not_a_map", "unexpected args type for set_model_response: string")
	})
}
