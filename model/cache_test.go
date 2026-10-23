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

package model_test

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/model"
)

func TestContextCacheConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     *model.ContextCacheConfig
		wantErr bool
	}{
		{name: "nil disables caching", cfg: nil},
		{name: "zero value takes the defaults", cfg: &model.ContextCacheConfig{}},
		{name: "every field set", cfg: &model.ContextCacheConfig{CacheIntervals: 5, TTL: time.Hour, MinTokens: 4096}},
		{name: "one interval", cfg: &model.ContextCacheConfig{CacheIntervals: 1}},
		{name: "most intervals", cfg: &model.ContextCacheConfig{CacheIntervals: 100}},
		{name: "too many intervals", cfg: &model.ContextCacheConfig{CacheIntervals: 101}, wantErr: true},
		{name: "negative intervals", cfg: &model.ContextCacheConfig{CacheIntervals: -1}, wantErr: true},
		{name: "one second TTL", cfg: &model.ContextCacheConfig{TTL: time.Second}},
		{name: "TTL under a second", cfg: &model.ContextCacheConfig{TTL: time.Second - time.Nanosecond}, wantErr: true},
		{name: "negative TTL", cfg: &model.ContextCacheConfig{TTL: -time.Second}, wantErr: true},
		{name: "negative MinTokens", cfg: &model.ContextCacheConfig{MinTokens: -1}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.Validate()
			if gotErr := err != nil; gotErr != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCacheMetadata_Active(t *testing.T) {
	tests := []struct {
		name string
		m    *model.CacheMetadata
		want bool
	}{
		{name: "nil", m: nil, want: false},
		{name: "fingerprint only", m: &model.CacheMetadata{Fingerprint: "abc", ContentsCount: 2}, want: false},
		{name: "active", m: &model.CacheMetadata{CacheName: "cachedContents/1", Fingerprint: "abc"}, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.m.Active(); got != tt.want {
				t.Errorf("Active() = %v, want %v", got, tt.want)
			}
		})
	}
}

// The wire forms below are what adk-python's LlmResponse writes for the same
// values, from model_dump_json(exclude_none=True, by_alias=True).
const (
	pythonActiveCacheJSON      = `{"cache_name":"cachedContents/1","expire_time":1767270600.5,"fingerprint":"abc","invocations_used":2,"contents_count":4,"created_at":1767268800.25}`
	pythonFingerprintOnlyJSON  = `{"fingerprint":"abc","contents_count":0}`
	pythonResponseWithCacheKey = `"cacheMetadata":{"fingerprint":"abc","contents_count":0}`
)

var activeCache = model.CacheMetadata{
	CacheName:       "cachedContents/1",
	ExpireTime:      time.Date(2026, 1, 1, 12, 30, 0, 500*int(time.Millisecond), time.UTC),
	InvocationsUsed: 2,
	CreatedAt:       time.Date(2026, 1, 1, 12, 0, 0, 250*int(time.Millisecond), time.UTC),
	Fingerprint:     "abc",
	ContentsCount:   4,
}

func TestCacheMetadata_MarshalJSON(t *testing.T) {
	tests := []struct {
		name string
		m    model.CacheMetadata
		want string
	}{
		{name: "active cache", m: activeCache, want: pythonActiveCacheJSON},
		// adk-python has no default for contents_count, so it is written even
		// when zero.
		{name: "fingerprint only", m: model.CacheMetadata{Fingerprint: "abc"}, want: pythonFingerprintOnlyJSON},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := json.Marshal(tt.m)
			if err != nil {
				t.Fatalf("json.Marshal() error = %v", err)
			}
			if string(got) != tt.want {
				t.Errorf("json.Marshal() = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestCacheMetadata_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want model.CacheMetadata
	}{
		{name: "active cache from adk-python", in: pythonActiveCacheJSON, want: activeCache},
		{name: "fingerprint only from adk-python", in: pythonFingerprintOnlyJSON, want: model.CacheMetadata{Fingerprint: "abc"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got model.CacheMetadata
			if err := json.Unmarshal([]byte(tt.in), &got); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("json.Unmarshal() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestLLMResponse_CacheMetadataJSON checks the field's key on a response,
// which adk-python's camelCase LlmResponse names cacheMetadata.
func TestLLMResponse_CacheMetadataJSON(t *testing.T) {
	encoded, err := json.Marshal(&model.LLMResponse{CacheMetadata: &model.CacheMetadata{Fingerprint: "abc"}})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	if got, want := string(encoded), "{"+pythonResponseWithCacheKey+"}"; got != want {
		t.Errorf("json.Marshal() = %s, want %s", got, want)
	}

	var decoded model.LLMResponse
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if diff := cmp.Diff(&model.CacheMetadata{Fingerprint: "abc"}, decoded.CacheMetadata); diff != "" {
		t.Errorf("round trip mismatch (-want +got):\n%s", diff)
	}

	empty, err := json.Marshal(&model.LLMResponse{})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	if string(empty) != "{}" {
		t.Errorf("a response without cache metadata encodes as %s, want {}", empty)
	}
}
