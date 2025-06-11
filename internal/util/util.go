// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package util

import (
	"cmp"
	"fmt"
	"iter"
	"reflect"
	"slices"
	"strings"
)

// Helpers below are copied from gopls' moremaps package.

// Sorted returns an iterator over the entries of m in key order.
func Sorted[M ~map[K]V, K cmp.Ordered, V any](m M) iter.Seq2[K, V] {
	// TODO(adonovan): use maps.Sorted if proposal #68598 is accepted.
	return func(yield func(K, V) bool) {
		keys := KeySlice(m)
		slices.Sort(keys)
		for _, k := range keys {
			if !yield(k, m[k]) {
				break
			}
		}
	}
}

// KeySlice returns the keys of the map M, like slices.Collect(maps.Keys(m)).
func KeySlice[M ~map[K]V, K comparable, V any](m M) []K {
	r := make([]K, 0, len(m))
	for k := range m {
		r = append(r, k)
	}
	return r
}

type JSONInfo struct {
	Omit     bool            // unexported or first tag element is "-"
	Name     string          // Go field name or first tag element. Empty if Omit is true.
	Settings map[string]bool // "omitempty", "omitzero", etc.
}

// FieldJSONInfo reports information about how encoding/json
// handles the given struct field.
// If the field is unexported, JSONInfo.Omit is true and no other JSONInfo field
// is populated.
// If the field is exported and has no tag, then Name is the field's name and all
// other fields are false.
// Otherwise, the information is obtained from the tag.
func FieldJSONInfo(f reflect.StructField) JSONInfo {
	if !f.IsExported() {
		return JSONInfo{Omit: true}
	}
	info := JSONInfo{Name: f.Name}
	if tag, ok := f.Tag.Lookup("json"); ok {
		name, rest, found := strings.Cut(tag, ",")
		// "-" means omit, but "-," means the name is "-"
		if name == "-" && !found {
			return JSONInfo{Omit: true}
		}
		if name != "" {
			info.Name = name
		}
		if len(rest) > 0 {
			info.Settings = map[string]bool{}
			for _, s := range strings.Split(rest, ",") {
				info.Settings[s] = true
			}
		}
	}
	return info
}

// Wrapf wraps *errp with the given formatted message if *errp is not nil.
func Wrapf(errp *error, format string, args ...any) {
	if *errp != nil {
		*errp = fmt.Errorf("%s: %w", fmt.Sprintf(format, args...), *errp)
	}
}
