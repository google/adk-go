// Copyright 2025 The A2A Authors
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

package a2asrv

import (
	"slices"
	"strings"
)

// RequestMeta holds the metadata associated with a request, like auth headers and signatures.
// Custom transport implementations can call WithCallContext to make it accessible during request processing.
type RequestMeta struct {
	kv map[string][]string
}

// NewRequestMeta is a [RequestMeta] constructor function .
func NewRequestMeta(src map[string][]string) *RequestMeta {
	if src == nil {
		return &RequestMeta{kv: map[string][]string{}}
	}

	kv := make(map[string][]string, len(src))
	for k, v := range src {
		kv[strings.ToLower(k)] = slices.Clone(v)
	}
	return &RequestMeta{kv: kv}
}

// Get performs a case-insensitive lookup of values for the given key.
func (rm *RequestMeta) Get(key string) ([]string, bool) {
	if rm == nil {
		return nil, false
	}

	val, ok := rm.kv[strings.ToLower(key)]
	return val, ok
}
