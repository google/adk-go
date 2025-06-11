// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package util

import (
	"reflect"
	"testing"
)

func TestJSONInfo(t *testing.T) {
	type S struct {
		A int
		B int `json:","`
		C int `json:"-"`
		D int `json:"-,"`
		E int `json:"echo"`
		F int `json:"foxtrot,omitempty"`
		g int `json:"golf"`
	}
	want := []JSONInfo{
		{Name: "A"},
		{Name: "B"},
		{Omit: true},
		{Name: "-"},
		{Name: "echo"},
		{Name: "foxtrot", Settings: map[string]bool{"omitempty": true}},
		{Omit: true},
	}
	tt := reflect.TypeFor[S]()
	for i := range tt.NumField() {
		got := FieldJSONInfo(tt.Field(i))
		if !reflect.DeepEqual(got, want[i]) {
			t.Errorf("got %+v, want %+v", got, want[i])
		}
	}
}
