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

package model

import (
	"encoding/json"
	"fmt"
	"math"
	"time"

	"google.golang.org/genai"
)

// maxCacheIntervals is the largest CacheIntervals a [ContextCacheConfig]
// accepts, matching adk-python's ContextCacheConfig.
const maxCacheIntervals = 100

// ContextCacheConfig turns on explicit context caching for the model requests
// of every LLM agent a runner drives. A nil config, the default, leaves caching
// off.
//
// With caching on, the system instruction, the tools and the stable prefix of
// the conversation are stored once as a cache on the model provider, and later
// requests name that cache instead of resending what it holds. A model that
// does not support caching ignores the config. The Gemini model in
// [google.golang.org/adk/v2/model/gemini] supports it, and its package
// documentation says when a cache is created and reused.
//
// The bookkeeping that carries a cache from one request to the next is the
// [CacheMetadata] on each model response, which the session stores with the
// event. The in-memory, database and Vertex AI session services keep it. A
// session service that drops it still serves the cache within one invocation,
// but the next invocation cannot find the cache to reuse or delete it, so it
// stays on the provider until its TTL runs out.
//
// A failure to create or delete a cache is not reported. Caching only saves
// tokens and time, so the request goes ahead without a cache instead.
//
// The zero value of each field selects its default.
type ContextCacheConfig struct {
	// CacheIntervals is the largest number of invocations that may use one
	// cache before it is replaced with a fresh one, between 1 and 100. Zero
	// means 10.
	CacheIntervals int
	// TTL is how long a cache lives on the provider after it is created, at
	// least one second. Zero means 30 minutes.
	TTL time.Duration
	// MinTokens is the smallest prompt, in tokens, that the agent's previous
	// request must have reached before a cache is created. It is compared with
	// that request's reported prompt token count, not an estimate of the next
	// one. Raise it to keep small prompts uncached, where storing a cache can
	// cost more than it saves. Zero means no minimum beyond the model's own.
	MinTokens int32
	// CreateHTTPOptions, when non-nil, apply to the call that creates a cache,
	// for example a Timeout. A creation that fails or times out leaves the
	// request uncached. One that times out after the provider has made the
	// cache leaves it there until its TTL runs out.
	CreateHTTPOptions *genai.HTTPOptions
}

// Validate reports whether c is a usable config. A nil config is valid and
// turns caching off.
func (c *ContextCacheConfig) Validate() error {
	if c == nil {
		return nil
	}
	if c.CacheIntervals < 0 || c.CacheIntervals > maxCacheIntervals {
		return fmt.Errorf("CacheIntervals must be between 1 and %d, or 0 for the default, got %d", maxCacheIntervals, c.CacheIntervals)
	}
	if c.TTL < 0 || (c.TTL > 0 && c.TTL < time.Second) {
		return fmt.Errorf("TTL must be at least one second, or 0 for the default, got %v", c.TTL)
	}
	if c.MinTokens < 0 {
		return fmt.Errorf("MinTokens must not be negative, got %d", c.MinTokens)
	}
	return nil
}

// CacheMetadata records the context cache behind a model response, so that the
// agent's next request can reuse it.
//
// It is in one of two states. An active cache has a CacheName, and every field
// describes that cache. Without a CacheName no cache exists yet: only
// Fingerprint and ContentsCount are set, and they describe the prefix the next
// request has to share before a cache is created for it.
//
// Token counts are not repeated here. Read them from the response's
// UsageMetadata, whose CachedContentTokenCount reports what the cache served.
//
// Its JSON encoding is adk-python's: snake_case keys, and times as seconds
// since the Unix epoch. Sessions shared with adk-python therefore read and
// write the same records.
type CacheMetadata struct {
	// CacheName is the provider's resource name for the cache, for example
	// "projects/123/locations/us-central1/cachedContents/456". Empty when no
	// cache is active.
	CacheName string
	// ExpireTime is when the provider deletes the cache. Zero when no cache is
	// active.
	ExpireTime time.Time
	// InvocationsUsed counts the invocations that have used the cache,
	// including the one that created it. Zero when no cache is active.
	InvocationsUsed int
	// CreatedAt is when the cache was created. Zero when no cache is active.
	CreatedAt time.Time
	// Fingerprint identifies the cacheable prefix, so that a request whose
	// prefix differs does not use the cache. It is opaque, and comparable only
	// with another fingerprint from the same model implementation.
	Fingerprint string
	// ContentsCount is how many leading contents the cache holds when one is
	// active, or the length of the prefix the fingerprint covers when none is.
	ContentsCount int
}

// Active reports whether m describes an existing cache rather than only a
// fingerprinted prefix. It is false for a nil m.
func (m *CacheMetadata) Active() bool {
	return m != nil && m.CacheName != ""
}

// cacheMetadataJSON is the wire form of [CacheMetadata], in adk-python's
// field names. Pointers leave out what is unset, as adk-python's
// exclude_none dump does. ContentsCount has no default there, so it is always
// written.
type cacheMetadataJSON struct {
	CacheName       string   `json:"cache_name,omitempty"`
	ExpireTime      *float64 `json:"expire_time,omitempty"`
	Fingerprint     string   `json:"fingerprint"`
	InvocationsUsed *int     `json:"invocations_used,omitempty"`
	ContentsCount   int      `json:"contents_count"`
	CreatedAt       *float64 `json:"created_at,omitempty"`
}

// MarshalJSON encodes m in adk-python's form.
func (m CacheMetadata) MarshalJSON() ([]byte, error) {
	out := cacheMetadataJSON{
		CacheName:     m.CacheName,
		ExpireTime:    epochSeconds(m.ExpireTime),
		Fingerprint:   m.Fingerprint,
		ContentsCount: m.ContentsCount,
		CreatedAt:     epochSeconds(m.CreatedAt),
	}
	if m.InvocationsUsed != 0 {
		out.InvocationsUsed = &m.InvocationsUsed
	}
	return json.Marshal(out)
}

// UnmarshalJSON decodes the form [CacheMetadata.MarshalJSON] writes, which is
// also the form adk-python writes.
func (m *CacheMetadata) UnmarshalJSON(data []byte) error {
	var in cacheMetadataJSON
	if err := json.Unmarshal(data, &in); err != nil {
		return err
	}
	*m = CacheMetadata{
		CacheName:     in.CacheName,
		ExpireTime:    fromEpochSeconds(in.ExpireTime),
		Fingerprint:   in.Fingerprint,
		ContentsCount: in.ContentsCount,
		CreatedAt:     fromEpochSeconds(in.CreatedAt),
	}
	if in.InvocationsUsed != nil {
		m.InvocationsUsed = *in.InvocationsUsed
	}
	return nil
}

func epochSeconds(t time.Time) *float64 {
	if t.IsZero() {
		return nil
	}
	// Whole seconds and the fraction are converted apart, as adk-python's
	// time.time() float holds them. Dividing the nanosecond count instead
	// loses the fraction's low digits to rounding.
	s := float64(t.Unix()) + float64(t.Nanosecond())/float64(time.Second)
	return &s
}

func fromEpochSeconds(s *float64) time.Time {
	if s == nil {
		return time.Time{}
	}
	whole, frac := math.Modf(*s)
	// Microseconds are all a float64 of today's epoch seconds holds exactly.
	return time.Unix(int64(whole), int64(math.Round(frac*1e6))*int64(time.Microsecond)).UTC()
}
