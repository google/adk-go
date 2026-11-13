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

package setmodelresponsetool

import (
	"cmp"
	"reflect"
	"strings"

	"google.golang.org/genai"
)

// A union is checked as pydantic's smart mode checks a Union field: every arm
// is tried, and the valid arm that sets the most record fields, then matches
// most exactly, then comes first, gives the answer.

// isUnion reports whether s is checked as a union: it has AnyOf arms and no
// type of its own.
func isUnion(s *genai.Schema) bool {
	t := upperType(s)
	return len(s.AnyOf) > 0 && (t == "" || t == genai.TypeUnspecified)
}

// unionArms returns the arms of the union s that take a non-null value, with
// nested unions flattened as Python flattens Union[int, Union[str, None]]. It
// reports whether s accepts a null: through a NULL arm or a nullable arm. A
// nil arm is untyped.
func unionArms(s *genai.Schema) (arms []*genai.Schema, nullable bool) {
	for _, arm := range s.AnyOf {
		switch {
		case arm == nil:
			arms = append(arms, nil)
			continue
		case upperType(arm) == genai.TypeNULL:
			nullable = true
		case isUnion(arm):
			inner, _ := unionArms(arm)
			arms = append(arms, inner...)
		default:
			arms = append(arms, arm)
		}
		nullable = nullable || isNullable(arm)
	}
	return arms, nullable
}

// normalizeUnion checks v, which is not a null that s accepts, against the
// arms of the union s.
func normalizeUnion(v any, s *genai.Schema, loc []string, issues *[]issue) any {
	arms, _ := unionArms(s)
	switch len(arms) {
	case 0:
		addIssue(issues, loc, "none_required", "Input should be None")
		return v
	case 1:
		// pydantic checks Optional[X] as X, so the issues carry no arm label.
		return normalize(v, arms[0], loc, issues)
	}
	out, _, failures := pickArm(v, arms, loc)
	*issues = append(*issues, failures...)
	return out
}

// pickArm returns the answer of the valid arm that pydantic picks, and its
// grade. When no arm is valid it returns v and every arm's issues, each under
// the arm's label at loc.
func pickArm(v any, arms []*genai.Schema, loc []string) (any, grade, []issue) {
	var failures []issue
	var out any
	var best grade
	found := false
	for _, arm := range arms {
		var armIssues []issue
		armOut := normalize(v, arm, child(loc, armLabel(arm)), &armIssues)
		if len(armIssues) > 0 {
			failures = append(failures, armIssues...)
			continue
		}
		if g := gradeOf(v, arm); !found || g.beats(best) {
			out, best, found = armOut, g, true
		}
	}
	if !found {
		return v, best, failures
	}
	return out, best, nil
}

// level is how closely a valid value matches an arm, as pydantic grades it.
type level int

const (
	lax    level = iota // a conversion that strict mode refuses
	strict              // a conversion that strict mode allows
	exact               // no conversion
)

// grade ranks a valid arm. fields is the number of the record's properties
// that the value sets, or -1 when the arm is not a record.
type grade struct {
	level  level
	fields int
}

// beats reports whether an arm graded g replaces the best arm so far: between
// two records the one that sets more fields wins, otherwise the higher level.
// A tie keeps the earlier arm.
func (g grade) beats(best grade) bool {
	if g.fields >= 0 && best.fields >= 0 && g.fields != best.fields {
		return g.fields > best.fields
	}
	return g.level > best.level
}

func gradeOf(v any, arm *genai.Schema) grade {
	g := grade{level: exactness(v, arm), fields: -1}
	if isRecord(arm) {
		m, _ := v.(map[string]any)
		g.fields = 0
		for name := range arm.Properties {
			if _, ok := m[name]; ok {
				g.fields++
			}
		}
	}
	return g
}

// exactness grades v, which is valid for s, as pydantic grades the input of a
// union arm. A record or a map read from a JSON object is strict at best, an
// untyped schema is strict as pydantic's Any is, and an enum read from a
// string is lax. Any number is exact for INTEGER and NUMBER, because a decoded
// answer holds every JSON number as a float64.
func exactness(v any, s *genai.Schema) level {
	if v == nil {
		return exact
	}
	if s == nil {
		return strict
	}
	if isUnion(s) {
		arms, _ := unionArms(s)
		_, g, _ := pickArm(v, arms, nil)
		return g.level
	}
	switch upperType(s) {
	case genai.TypeString:
		if len(s.Enum) > 0 {
			return lax
		}
		return exact
	case genai.TypeInteger, genai.TypeNumber:
		switch v.(type) {
		case string, bool:
			return lax
		}
		return exact
	case genai.TypeBoolean:
		if _, ok := v.(bool); !ok {
			return lax
		}
		return exact
	case genai.TypeArray:
		l := exact
		rv := reflect.ValueOf(v)
		for i := range rv.Len() {
			l = min(l, exactness(rv.Index(i).Interface(), s.Items))
		}
		return l
	case genai.TypeObject:
		l := strict
		m, _ := v.(map[string]any)
		for name, prop := range s.Properties {
			if val, ok := m[name]; ok {
				l = min(l, exactness(val, prop))
			}
		}
		return l
	}
	return strict
}

// armLabel names s in an issue's location as pydantic names a union member.
func armLabel(s *genai.Schema) string {
	if s == nil {
		return "any"
	}
	if isUnion(s) {
		arms, _ := unionArms(s)
		labels := make([]string, len(arms))
		for i, arm := range arms {
			labels[i] = armLabel(arm)
		}
		if len(labels) == 1 {
			return labels[0]
		}
		return "union[" + strings.Join(labels, ",") + "]"
	}
	switch upperType(s) {
	case genai.TypeString:
		if len(s.Enum) > 0 {
			return "str-enum[" + cmp.Or(s.Title, "Enum") + "]"
		}
		return "str"
	case genai.TypeInteger:
		return "int"
	case genai.TypeNumber:
		return "float"
	case genai.TypeBoolean:
		return "bool"
	case genai.TypeArray:
		item := armLabel(s.Items)
		if s.Items != nil && isNullable(s.Items) {
			item = "nullable[" + item + "]"
		}
		return "list[" + item + "]"
	case genai.TypeObject:
		if isRecord(s) {
			return titleOf(s)
		}
		return "dict[str,any]"
	}
	return "any"
}
