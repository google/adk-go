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

package openapitoolset

import "testing"

func TestSnakeCase(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"camelCase", "camel_case"},
		{"UpperCamelCase", "upper_camel_case"},
		{"space separated", "space_separated"},
		{"calendar.calendars.get", "calendar_calendars_get"},
		{"HTTPResponse", "http_response"},
		{"REST API", "rest_api"},
		{"calendarId", "calendar_id"},
		{"createUser", "create_user"},
		{"/test_get", "test_get"},
		{"already_snake", "already_snake"},
		{"", ""},
	}
	for _, tc := range tests {
		if got := snakeCase(tc.in); got != tc.want {
			t.Errorf("snakeCase(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestRenameKeyword(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"in", "param_in"},
		{"for", "param_for"},
		{"class", "param_class"},
		{"normal", "normal"},
		{"param_if", "param_if"},
		{"", ""},
	}
	for _, tc := range tests {
		if got := renameKeyword(tc.in); got != tc.want {
			t.Errorf("renameKeyword(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestTruncate(t *testing.T) {
	tests := []struct {
		in   string
		n    int
		want string
	}{
		{"hello", 3, "hel"},
		{"hello", 5, "hello"},
		{"hello", 10, "hello"},
		{"héllo", 2, "hé"}, // counted in code points, like Python
		{"", 3, ""},
	}
	for _, tc := range tests {
		if got := truncate(tc.in, tc.n); got != tc.want {
			t.Errorf("truncate(%q, %d) = %q, want %q", tc.in, tc.n, got, tc.want)
		}
	}
}
