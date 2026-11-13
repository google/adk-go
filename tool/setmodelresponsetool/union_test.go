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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

// The arms below mirror the pydantic types of the probes cited in each row.
var (
	intArm  = &genai.Schema{Type: genai.TypeInteger}
	numArm  = &genai.Schema{Type: genai.TypeNumber}
	strArm  = &genai.Schema{Type: genai.TypeString}
	boolArm = &genai.Schema{Type: genai.TypeBoolean}
	nullArm = &genai.Schema{Type: genai.TypeNULL}
	dictArm = &genai.Schema{Type: genai.TypeObject}
	addrArm = &genai.Schema{
		Type:       genai.TypeObject,
		Title:      "Addr",
		Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
		Required:   []string{"city"},
	}
	bArm = &genai.Schema{
		Type:  genai.TypeObject,
		Title: "B",
		Properties: map[string]*genai.Schema{
			"city": {Type: genai.TypeString},
			"zip":  {Type: genai.TypeInteger, Default: float64(0)},
		},
		Required: []string{"city"},
	}
	nArm = &genai.Schema{
		Type:       genai.TypeObject,
		Title:      "N",
		Properties: map[string]*genai.Schema{"n": {Type: genai.TypeInteger}},
		Required:   []string{"n"},
	}
	sArm = &genai.Schema{
		Type:       genai.TypeObject,
		Title:      "S",
		Properties: map[string]*genai.Schema{"n": {Type: genai.TypeString}},
		Required:   []string{"n"},
	}
	tagsArm = &genai.Schema{
		Type:  genai.TypeObject,
		Title: "Tags",
		Properties: map[string]*genai.Schema{
			"tags": {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}, Nullable: new(true)},
		},
	}
	oneArm = &genai.Schema{Type: genai.TypeString, Title: "One", Enum: []string{"1"}}
)

func union(arms ...*genai.Schema) *genai.Schema {
	return &genai.Schema{AnyOf: arms}
}

func listOf(items *genai.Schema) *genai.Schema {
	return &genai.Schema{Type: genai.TypeArray, Items: items}
}

// outWith returns a record "Out" whose only property, the required "v", has
// the schema s.
func outWith(s *genai.Schema) *genai.Schema {
	return &genai.Schema{
		Type:       genai.TypeObject,
		Title:      "Out",
		Properties: map[string]*genai.Schema{"v": s},
		Required:   []string{"v"},
	}
}

// TestRun_UnionFields_InvalidAnswerGetsFeedback fails if a field whose schema
// is an AnyOf is not checked against its arms: adk-python reports these three
// issues for v: int | str and home: Optional[Addr] (probe "F1 failing case").
func TestRun_UnionFields_InvalidAnswerGetsFeedback(t *testing.T) {
	tl := newTool(t, &genai.Schema{
		Type:             genai.TypeObject,
		Title:            "Out",
		PropertyOrdering: []string{"v", "home"},
		Properties: map[string]*genai.Schema{
			"v":    union(intArm, strArm),
			"home": union(addrArm, nullArm),
		},
		Required: []string{"v", "home"},
	})
	want := feedbackHead +
		"3 validation errors for Out\n" +
		"v.int\n  Input should be a valid integer [type=int_type]\n" +
		"v.str\n  Input should be a valid string [type=string_type]\n" +
		"home.city\n  Field required [type=missing]\n" +
		feedbackTail

	checkFeedback(t, tl, map[string]any{"v": []any{float64(1), float64(2)}, "home": map[string]any{"zip": float64(1)}}, want)
}

// TestRun_UnionField_PicksTheArmPydanticPicks fails if the union does not
// keep the valid arm with the most record fields set, then the most exact
// match, then the first arm, or if the picked arm's answer is not normalized.
func TestRun_UnionField_PicksTheArmPydanticPicks(t *testing.T) {
	tests := []struct {
		name   string
		schema *genai.Schema
		in     any
		want   any
	}{
		{"P1_exact_str_beats_lax_int", union(intArm, strArm), "25", "25"},
		{"P2_exact_int_after_failed_str", union(strArm, intArm), float64(1), float64(1)},
		{"P3_optional_record_drops_unknown_keys", union(addrArm, nullArm), map[string]any{"city": "x", "extra": float64(1)}, map[string]any{"city": "x"}},
		{"P8_lax_number_after_failed_int", union(intArm, numArm), "2.5", float64(2.5)},
		{"P10_lax_int_after_failed_str", union(strArm, intArm), true, float64(1)},
		{"P11_exact_bool_beats_lax_int", union(intArm, boolArm), true, true},
		{"P16_more_fields_set_wins", union(addrArm, bArm), map[string]any{"city": "x", "zip": float64(5)}, map[string]any{"city": "x", "zip": float64(5)}},
		{"P17_equal_fields_set_first_record_wins", union(bArm, addrArm), map[string]any{"city": "x"}, map[string]any{"city": "x", "zip": float64(0)}},
		{"P18_list_coerces_its_items", union(listOf(intArm), strArm), []any{"1", float64(2)}, []any{float64(1), float64(2)}},
		{"P19_list_takes_lowest_item_grade", union(listOf(intArm), listOf(strArm)), []any{"1"}, []any{"1"}},
		{"P21_exact_str_beats_lax_bool", union(boolArm, strArm), "true", "true"},
		{"P25_exact_int_beats_lax_bool", union(boolArm, intArm), float64(1), float64(1)},
		{"Q16_dict_beats_record_with_lax_field", union(nArm, dictArm), map[string]any{"n": "1"}, map[string]any{"n": "1"}},
		{"R5_untyped_beats_lax_int", union(intArm, &genai.Schema{}), "1", "1"},
		{"nil_arm_is_untyped", union(intArm, nil), "1", "1"},
		{"S1_enum_is_lax_so_first_lax_arm_wins", union(intArm, oneArm), "1", float64(1)},
		{"S2_enum_first_wins_a_tie", union(oneArm, intArm), "1", "1"},
		{"T1_record_and_dict_tie_record_first", union(addrArm, dictArm), map[string]any{"city": "x", "extra": float64(1)}, map[string]any{"city": "x"}},
		{"T2_record_and_dict_tie_dict_first", union(dictArm, addrArm), map[string]any{"city": "x", "extra": float64(1)}, map[string]any{"city": "x", "extra": float64(1)}},
		{"V2_equal_fields_set_higher_grade_wins", union(nArm, sArm), map[string]any{"n": "1"}, map[string]any{"n": "1"}},
		{"V3_equal_fields_set_first_wins", union(addrArm, bArm), map[string]any{"city": "x"}, map[string]any{"city": "x"}},
		{"V4_union_items_take_the_grade_of_the_picked_arm", union(listOf(union(intArm, boolArm)), listOf(strArm)), []any{"1"}, []any{"1"}},
		{"V5_record_and_untyped_tie_record_first", union(addrArm, &genai.Schema{}), map[string]any{"city": "x", "extra": float64(1)}, map[string]any{"city": "x"}},
		{"V5_record_and_nil_arm_tie_record_first", union(addrArm, nil), map[string]any{"city": "x", "extra": float64(1)}, map[string]any{"city": "x"}},
		{"W1_record_and_untyped_tie_untyped_first", union(&genai.Schema{}, addrArm), map[string]any{"city": "x", "extra": float64(1)}, map[string]any{"city": "x", "extra": float64(1)}},
		{"V6_record_arm_with_a_null_list_field", union(tagsArm, intArm), map[string]any{"tags": nil}, map[string]any{}},
		{"type_unspecified_is_a_union", &genai.Schema{Type: genai.TypeUnspecified, AnyOf: []*genai.Schema{strArm, intArm}}, true, float64(1)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tl := newTool(t, outWith(tc.schema))
			ctx := newToolContext(t)

			got, err := tl.Run(ctx, map[string]any{"v": tc.in})
			if err != nil {
				t.Fatalf("Run({v: %#v}) error: %v", tc.in, err)
			}
			if diff := cmp.Diff(map[string]any{"v": tc.want}, got); diff != "" {
				t.Errorf("Run({v: %#v}) mismatch (-want +got):\n%s", tc.in, diff)
			}
		})
	}
}

// TestRun_UnionField_NoArmMatches fails if a value that matches no arm does
// not get every arm's issues, in arm order, under pydantic's label of the arm.
func TestRun_UnionField_NoArmMatches(t *testing.T) {
	untitledRecord := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
		Required:   []string{"city"},
	}
	tests := []struct {
		name   string
		schema *genai.Schema
		in     any
		want   string // the feedback between feedbackHead and feedbackTail
	}{
		{
			"P7_null_without_null_arm", union(intArm, strArm), nil,
			"2 validation errors for Out\n" +
				"v.int\n  Input should be a valid integer [type=int_type]\n" +
				"v.str\n  Input should be a valid string [type=string_type]\n",
		},
		{
			"P13_record_arm_label_is_its_title", union(addrArm, intArm),
			map[string]any{"zip": float64(1)},
			"2 validation errors for Out\n" +
				"v.Addr.city\n  Field required [type=missing]\n" +
				"v.int\n  Input should be a valid integer [type=int_type]\n",
		},
		{
			"untitled_record_arm", union(untitledRecord, intArm),
			map[string]any{},
			"2 validation errors for Out\n" +
				"v.object.city\n  Field required [type=missing]\n" +
				"v.int\n  Input should be a valid integer [type=int_type]\n",
		},
		{
			"P14_null_arm_keeps_the_other_labels", union(intArm, strArm, nullArm),
			[]any{float64(1)},
			"2 validation errors for Out\n" +
				"v.int\n  Input should be a valid integer [type=int_type]\n" +
				"v.str\n  Input should be a valid string [type=string_type]\n",
		},
		{
			"Q8_union_items_of_a_list", listOf(union(intArm, strArm)),
			[]any{"1", float64(2), []any{float64(3)}},
			"2 validation errors for Out\n" +
				"v.2.int\n  Input should be a valid integer [type=int_type]\n" +
				"v.2.str\n  Input should be a valid string [type=string_type]\n",
		},
		{
			"Q9_optional_reports_its_arm_without_label", union(intArm, nullArm), "x",
			"1 validation error for Out\n" +
				"v\n  Input should be a valid integer, unable to parse string as an integer [type=int_parsing]\n",
		},
		{
			"Q11_list_of_records_arm", union(intArm, listOf(addrArm)),
			[]any{map[string]any{"zip": float64(1)}},
			"2 validation errors for Out\n" +
				"v.int\n  Input should be a valid integer [type=int_type]\n" +
				"v.list[Addr].0.city\n  Field required [type=missing]\n",
		},
		{
			"Q13_null_only", union(nullArm), float64(1),
			"1 validation error for Out\n" +
				"v\n  Input should be None [type=none_required]\n",
		},
		{
			"R1_enum_arm", union(&genai.Schema{Type: genai.TypeString, Title: "Color", Enum: []string{"a", "b"}}, intArm), "z",
			"2 validation errors for Out\n" +
				"v.str-enum[Color]\n  Input should be 'a' or 'b' [type=enum]\n" +
				"v.int\n  Input should be a valid integer, unable to parse string as an integer [type=int_parsing]\n",
		},
		{
			"untitled_enum_arm", union(&genai.Schema{Type: genai.TypeString, Enum: []string{"a"}}, intArm), "z",
			"2 validation errors for Out\n" +
				"v.str-enum[Enum]\n  Input should be 'a' [type=enum]\n" +
				"v.int\n  Input should be a valid integer, unable to parse string as an integer [type=int_parsing]\n",
		},
		{
			"R2_dict_arm", union(dictArm, intArm), "x",
			"2 validation errors for Out\n" +
				"v.dict[str,any]\n  Input should be a valid dictionary [type=dict_type]\n" +
				"v.int\n  Input should be a valid integer, unable to parse string as an integer [type=int_parsing]\n",
		},
		{
			"R3_list_of_nullable_items", union(listOf(&genai.Schema{Type: genai.TypeInteger, Nullable: new(true)}), strArm),
			[]any{"x"},
			"2 validation errors for Out\n" +
				"v.list[nullable[int]].0\n  Input should be a valid integer, unable to parse string as an integer [type=int_parsing]\n" +
				"v.str\n  Input should be a valid string [type=string_type]\n",
		},
		{
			"list_of_optional_items_written_as_union", union(listOf(union(intArm, nullArm)), strArm),
			[]any{"x"},
			"2 validation errors for Out\n" +
				"v.list[nullable[int]].0\n  Input should be a valid integer, unable to parse string as an integer [type=int_parsing]\n" +
				"v.str\n  Input should be a valid string [type=string_type]\n",
		},
		{
			"R4_list_of_untyped_items", union(&genai.Schema{Type: genai.TypeArray}, intArm), "x",
			"2 validation errors for Out\n" +
				"v.list[any]\n  Input should be a valid list [type=list_type]\n" +
				"v.int\n  Input should be a valid integer, unable to parse string as an integer [type=int_parsing]\n",
		},
		{
			"R6_list_of_union_items", union(listOf(union(intArm, strArm)), intArm),
			[]any{[]any{float64(1)}},
			"3 validation errors for Out\n" +
				"v.list[union[int,str]].0.int\n  Input should be a valid integer [type=int_type]\n" +
				"v.list[union[int,str]].0.str\n  Input should be a valid string [type=string_type]\n" +
				"v.int\n  Input should be a valid integer [type=int_type]\n",
		},
		{
			"T3_list_of_nullable_union_items", union(listOf(union(intArm, strArm, nullArm)), intArm),
			[]any{[]any{float64(1)}},
			"3 validation errors for Out\n" +
				"v.list[nullable[union[int,str]]].0.int\n  Input should be a valid integer [type=int_type]\n" +
				"v.list[nullable[union[int,str]]].0.str\n  Input should be a valid string [type=string_type]\n" +
				"v.int\n  Input should be a valid integer [type=int_type]\n",
		},
		{
			"T4_nested_union_is_flattened", union(intArm, union(boolArm, nullArm)),
			[]any{},
			"2 validation errors for Out\n" +
				"v.int\n  Input should be a valid integer [type=int_type]\n" +
				"v.bool\n  Input should be a valid boolean [type=bool_type]\n",
		},
		{
			"V1_nested_union_is_flattened", union(intArm, union(boolArm, strArm, nullArm)),
			[]any{},
			"3 validation errors for Out\n" +
				"v.int\n  Input should be a valid integer [type=int_type]\n" +
				"v.bool\n  Input should be a valid boolean [type=bool_type]\n" +
				"v.str\n  Input should be a valid string [type=string_type]\n",
		},
		{
			"V7_list_of_empty_schema_items", union(listOf(&genai.Schema{}), intArm), "x",
			"2 validation errors for Out\n" +
				"v.list[any]\n  Input should be a valid list [type=list_type]\n" +
				"v.int\n  Input should be a valid integer, unable to parse string as an integer [type=int_parsing]\n",
		},
		{
			"T6_number_arm", union(numArm, boolArm),
			[]any{},
			"2 validation errors for Out\n" +
				"v.float\n  Input should be a valid number [type=float_type]\n" +
				"v.bool\n  Input should be a valid boolean [type=bool_type]\n",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tl := newTool(t, outWith(tc.schema))

			checkFeedback(t, tl, map[string]any{"v": tc.in}, feedbackHead+tc.want+feedbackTail)
		})
	}
}

// TestRun_UnionField_Null fails if a NULL arm, a nullable arm or a NULL arm of
// a nested union does not accept a null, or if such a null does not follow the
// object rule for nulls: kept for a required key, dropped otherwise.
func TestRun_UnionField_Null(t *testing.T) {
	schemas := map[string]*genai.Schema{
		"null_arm":        union(addrArm, nullArm),
		"nullable_arm":    union(&genai.Schema{Type: genai.TypeInteger, Nullable: new(true)}, strArm),
		"nested_null_arm": union(intArm, union(boolArm, nullArm)),
	}
	for name, s := range schemas {
		t.Run(name, func(t *testing.T) {
			tl := newTool(t, &genai.Schema{
				Type:       genai.TypeObject,
				Properties: map[string]*genai.Schema{"req": s, "opt": s},
				Required:   []string{"req"},
			})
			ctx := newToolContext(t)

			got, err := tl.Run(ctx, map[string]any{"req": nil, "opt": nil})
			if err != nil {
				t.Fatalf("Run() error: %v", err)
			}
			if diff := cmp.Diff(map[string]any{"req": nil}, got); diff != "" {
				t.Errorf("Run() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestRun_TypedSchemaWithAnyOf_IsCheckedByItsType fails if a schema that has
// both a type and AnyOf arms is checked as a union: its type decides, as it
// did before unions were checked.
func TestRun_TypedSchemaWithAnyOf_IsCheckedByItsType(t *testing.T) {
	tl := newTool(t, outWith(&genai.Schema{Type: genai.TypeInteger, AnyOf: []*genai.Schema{strArm, nullArm}}))
	tests := []struct {
		in   any
		want string
	}{
		{nil, "v\n  Input should be a valid integer [type=int_type]\n"},
		{"x", "v\n  Input should be a valid integer, unable to parse string as an integer [type=int_parsing]\n"},
	}
	for _, tc := range tests {
		t.Run(fmt.Sprintf("%#v", tc.in), func(t *testing.T) {
			checkFeedback(t, tl, map[string]any{"v": tc.in}, feedbackHead+"1 validation error for Out\n"+tc.want+feedbackTail)
		})
	}
}

// record returns a record schema with the given title and properties, none of
// them required.
func record(title string, props map[string]*genai.Schema) *genai.Schema {
	return &genai.Schema{Type: genai.TypeObject, Title: title, Properties: props}
}

// defaultZero is an INTEGER property that defaults to 0.
func defaultZero() *genai.Schema {
	return &genai.Schema{Type: genai.TypeInteger, Default: float64(0)}
}

// TestRun_UnionField_CountsFieldsSetInNestedRecords fails if the fields-set
// comparison of a union counts only the arm's own fields, and not the fields
// set in the records nested in its properties, its list items and the picked
// arm of a nested union, added up over all of them. Each row is the pydantic
// probe it names.
func TestRun_UnionField_CountsFieldsSetInNestedRecords(t *testing.T) {
	inner := record("Inner", map[string]*genai.Schema{"p": defaultZero(), "q": defaultZero()})
	inner2 := record("Inner2", map[string]*genai.Schema{"p": defaultZero(), "r": defaultZero()})
	pInt := &genai.Schema{
		Type:       genai.TypeObject,
		Title:      "PI",
		Properties: map[string]*genai.Schema{"p": intArm},
		Required:   []string{"p"},
	}
	// bd is pydantic's BD(x: dict, y: int = 0): x holds a JSON object whose
	// values are untyped.
	bd := &genai.Schema{
		Type:       genai.TypeObject,
		Title:      "BD",
		Properties: map[string]*genai.Schema{"x": dictArm, "y": defaultZero()},
		Required:   []string{"x"},
	}
	withX := func(title string, x *genai.Schema) *genai.Schema {
		return &genai.Schema{
			Type:       genai.TypeObject,
			Title:      title,
			Properties: map[string]*genai.Schema{"x": x},
			Required:   []string{"x"},
		}
	}
	a := withX("A", inner)
	deep := withX("AD", &genai.Schema{
		Type:       genai.TypeObject,
		Title:      "Deep1",
		Properties: map[string]*genai.Schema{"d": record("Deep2", map[string]*genai.Schema{"k": defaultZero(), "m": defaultZero()})},
		Required:   []string{"d"},
	})
	w1 := &genai.Schema{
		Type:       genai.TypeObject,
		Title:      "W1",
		Properties: map[string]*genai.Schema{"v": union(intArm, addrArm)},
		Required:   []string{"v"},
	}
	w2 := &genai.Schema{
		Type:       genai.TypeObject,
		Title:      "W2",
		Properties: map[string]*genai.Schema{"v": dictArm, "w": defaultZero()},
		Required:   []string{"v"},
	}
	ld := &genai.Schema{
		Type:       genai.TypeObject,
		Title:      "LD",
		Properties: map[string]*genai.Schema{"v": {Type: genai.TypeArray}, "n": defaultZero()},
		Required:   []string{"v"},
	}
	la := &genai.Schema{
		Type:       genai.TypeObject,
		Title:      "LA",
		Properties: map[string]*genai.Schema{"v": listOf(addrArm)},
		Required:   []string{"v"},
	}
	two := record("Two", map[string]*genai.Schema{"a": defaultZero(), "b": defaultZero()})
	one := record("One", map[string]*genai.Schema{"a": defaultZero(), "c": defaultZero()})
	e := record("E", map[string]*genai.Schema{"a": defaultZero()})
	f := &genai.Schema{
		Type:       genai.TypeObject,
		Title:      "F",
		Properties: map[string]*genai.Schema{"b": intArm},
		Required:   []string{"b"},
	}

	tests := []struct {
		name   string
		schema *genai.Schema
		in     any
		want   any
	}{
		{
			"N1a_list_items_count", union(listOf(addrArm), listOf(bArm)),
			[]any{map[string]any{"city": "x", "zip": float64(5)}},
			[]any{map[string]any{"city": "x", "zip": float64(5)}},
		},
		{
			"N1c_record_in_a_property_counts_and_dict_values_do_not", union(bd, a),
			map[string]any{"x": map[string]any{"p": float64(1), "q": float64(2), "r": float64(3)}},
			map[string]any{"x": map[string]any{"p": float64(1), "q": float64(2)}},
		},
		{
			"N1f_equal_counts_in_list_items_first_wins", union(listOf(addrArm), listOf(bArm)),
			[]any{map[string]any{"city": "x"}},
			[]any{map[string]any{"city": "x"}},
		},
		{
			"N1g_equal_counts_in_nested_records_first_wins", union(withX("P", inner), withX("Q", inner2)),
			map[string]any{"x": map[string]any{"p": float64(1)}},
			map[string]any{"x": map[string]any{"p": float64(1), "q": float64(0)}},
		},
		{
			"N1i_record_two_levels_down_counts", union(bd, deep),
			map[string]any{"x": map[string]any{"d": map[string]any{"k": float64(1), "m": float64(2)}}},
			map[string]any{"x": map[string]any{"d": map[string]any{"k": float64(1), "m": float64(2)}}},
		},
		{
			"N1l_picked_arm_of_a_nested_union_counts", union(w2, w1),
			map[string]any{"v": map[string]any{"city": "x"}},
			map[string]any{"v": map[string]any{"city": "x"}},
		},
		{
			"N1n_count_beats_a_higher_grade", union(bd, withX("Wb", pInt)),
			map[string]any{"x": map[string]any{"p": "1"}},
			map[string]any{"x": map[string]any{"p": float64(1)}},
		},
		{
			"N1o_optional_record_arm_counts", union(bd, union(a, nullArm)),
			map[string]any{"x": map[string]any{"p": float64(1), "q": float64(2)}},
			map[string]any{"x": map[string]any{"p": float64(1), "q": float64(2)}},
		},
		{
			// Two sets 1+1+2 fields and One 2+2+1: One wins only by the sum.
			"N3d_counts_add_up_over_items", union(listOf(two), listOf(one)),
			[]any{
				map[string]any{"a": float64(1), "c": float64(1)},
				map[string]any{"a": float64(1), "c": float64(1)},
				map[string]any{"a": float64(1), "b": float64(1)},
			},
			[]any{
				map[string]any{"a": float64(1), "c": float64(1)},
				map[string]any{"a": float64(1), "c": float64(1)},
				map[string]any{"a": float64(1), "c": float64(0)},
			},
		},
		{
			// E sets no field but still counts 0, so F's 1 beats it despite
			// F's lax grade.
			"N3a_record_item_setting_no_field_counts_zero", union(listOf(e), listOf(f)),
			[]any{map[string]any{"b": "1"}},
			[]any{map[string]any{"b": float64(1)}},
		},
		{
			"N3c_records_in_nested_lists_count", union(listOf(listOf(addrArm)), listOf(listOf(bArm))),
			[]any{[]any{map[string]any{"city": "x", "zip": float64(5)}}},
			[]any{[]any{map[string]any{"city": "x", "zip": float64(5)}}},
		},
		{
			"N2b_records_in_a_list_property_count", union(ld, la),
			map[string]any{"v": []any{map[string]any{"city": "x"}}},
			map[string]any{"v": []any{map[string]any{"city": "x"}}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tl := newTool(t, outWith(tc.schema))
			ctx := newToolContext(t)

			got, err := tl.Run(ctx, map[string]any{"v": tc.in})
			if err != nil {
				t.Fatalf("Run({v: %#v}) error: %v", tc.in, err)
			}
			if diff := cmp.Diff(map[string]any{"v": tc.want}, got); diff != "" {
				t.Errorf("Run({v: %#v}) mismatch (-want +got):\n%s", tc.in, diff)
			}
		})
	}
}

// TestRun_UnionField_ListOfUntypedItemsIsExact fails if a list whose items are
// untyped (pydantic's list[Any]) is graded below exact: pydantic does not
// check such items, so the list beats a list of records, while a list of
// nullable untyped items (list[Optional[Any]]) is checked and only strict.
func TestRun_UnionField_ListOfUntypedItemsIsExact(t *testing.T) {
	in := []any{map[string]any{"city": "x", "e": float64(1)}}
	tests := []struct {
		name   string
		schema *genai.Schema
		in     any
		want   any
	}{
		{"N2d_nil_items", union(listOf(addrArm), listOf(nil)), in, in},
		{"A2_empty_schema_items", union(listOf(addrArm), listOf(&genai.Schema{})), in, in},
		{"A2_type_unspecified_items", union(listOf(addrArm), listOf(&genai.Schema{Type: genai.TypeUnspecified})), in, in},
		{"A6_nested_lists", union(listOf(listOf(addrArm)), listOf(listOf(nil))), []any{in}, []any{in}},
		{
			"A1_nullable_untyped_items_tie_record_first",
			union(listOf(addrArm), listOf(&genai.Schema{Nullable: new(true)})),
			in,
			[]any{map[string]any{"city": "x"}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tl := newTool(t, outWith(tc.schema))
			ctx := newToolContext(t)

			got, err := tl.Run(ctx, map[string]any{"v": tc.in})
			if err != nil {
				t.Fatalf("Run({v: %#v}) error: %v", tc.in, err)
			}
			if diff := cmp.Diff(map[string]any{"v": tc.want}, got); diff != "" {
				t.Errorf("Run({v: %#v}) mismatch (-want +got):\n%s", tc.in, diff)
			}
		})
	}
}
