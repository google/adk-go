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

package gemini

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
)

// The defaults model.ContextCacheConfig documents for its zero fields.
const (
	defaultCacheIntervals = 10
	defaultCacheTTL       = 30 * time.Minute
)

// The smallest explicit cache each named Gemini model family accepts. For a
// tuned model or endpoint whose name does not say its family, the server is
// left to decide.
const (
	gemini25MinCacheTokens = 2048
	gemini3MinCacheTokens  = 4096
)

// cacheCleanupTimeout bounds deleting a cache once the request that decided
// to delete it has ended, possibly by cancellation.
const cacheCleanupTimeout = 30 * time.Second

// contextCache manages the explicit context caches of one model request,
// following adk-python's GeminiContextCacheManager
// (models/gemini_context_cache_manager.py).
//
// A cache holds the system instruction, the tools and their config, and a
// prefix of the contents: everything before the trailing run of user
// contents, which is the turn being answered. It is created only once two
// consecutive requests of an agent agree on that prefix, sized by what the
// earlier request cost, and only when the prefix reaches the model's minimum
// cache size: 2048 tokens for Gemini 2.5 and 4096 for Gemini 3. It is reused
// until it expires, has served CacheIntervals invocations, or the prefix
// changes. The state that carries it from one request to the next is the
// model.CacheMetadata on each response.
//
// It differs from adk-python in these ways, each deliberate:
//
//   - The cached fields are left on the caller's request. A copy without them
//     is sent, because the flow and its telemetry read the request after the
//     model call.
//   - A cache made unusable is deleted only once the response that replaces it
//     has reached the caller, and a cache created for a response that never
//     reaches the caller is deleted at once. adk-python deletes first, so a
//     failed call leaves the session naming a cache that no longer exists,
//     and a later request whose prefix matches it again is sent with it and
//     rejected.
//   - A request that already names a cache in its config is sent as it is.
//     adk-python replaces that cache with its own.
//   - Nothing is logged when a cache cannot be created or deleted, since
//     ADK Go library code has no logger. The request goes ahead uncached, as
//     in adk-python.
type contextCache struct {
	client *genai.Client
	// model is the name the request is sent to, which explicit caches are
	// specific to.
	model string
	// trackingHeader identifies ADK on the cache calls, as on the model calls.
	trackingHeader string
}

// cachePlan is what prepare decided for one request.
type cachePlan struct {
	// req is the request to send.
	req *model.LLMRequest
	// metadata is what to record on the response, nil when caching does not
	// apply to the request.
	metadata *model.CacheMetadata
	// created names a cache made for this request. It is deleted if the
	// response never reaches the caller, since nothing records it then.
	created string
	// stale names a cache this request replaced. It is deleted once the
	// response reaches the caller, and kept otherwise, because the session
	// still names it until a response replaces that record.
	stale string
}

// prepare decides which cache req uses, creating a cache on the provider when
// one is due.
func (c *contextCache) prepare(ctx context.Context, req *model.LLMRequest) cachePlan {
	if req.Config.CachedContent != "" {
		return cachePlan{req: req}
	}
	fp := newFingerprinter(req, c.model, scopeFor(c.client.ClientConfig()))
	old := req.CacheMetadata
	if old == nil {
		// No earlier prefix to compare against, so nothing is created yet.
		return cachePlan{req: req, metadata: fp.metadataFor(countContentsToCache(req.Contents))}
	}

	oldPrefix := fp.of(old.ContentsCount)
	if c.isValid(ctx, req, oldPrefix) {
		m := *old
		return cachePlan{req: applyCache(req, old.CacheName, old.ContentsCount), metadata: &m}
	}

	plan := cachePlan{req: req}
	if old.Active() {
		plan.stale = old.CacheName
	}
	if oldPrefix == "" || oldPrefix != old.Fingerprint {
		// The prefix changed. Start over from this request's prefix, which
		// leaves out its trailing user contents, such as a dynamic
		// instruction, so that they cannot break the next comparison.
		plan.metadata = fp.metadataFor(countContentsToCache(req.Contents))
		return plan
	}

	count := max(old.ContentsCount, countContentsToCache(req.Contents))
	created := c.createCache(ctx, req, count, fp.of(count))
	if created == nil {
		// No cache, for example because the prefix is still below the
		// model's minimum. Keep the longest stable prefix for the next
		// attempt.
		plan.metadata = fp.metadataFor(count)
		return plan
	}
	plan.req = applyCache(req, created.CacheName, count)
	plan.metadata = created
	plan.created = created.CacheName
	return plan
}

// isValid reports whether req can use the active cache its metadata names:
// the cache has not expired, has not served more than CacheIntervals
// invocations, and holds the prefix req still has, whose fingerprint is
// prefix.
func (c *contextCache) isValid(ctx context.Context, req *model.LLMRequest, prefix string) bool {
	m := req.CacheMetadata
	if !m.Active() {
		return false
	}
	if !platform.Now(ctx).Before(m.ExpireTime) {
		return false
	}
	if m.InvocationsUsed > cacheIntervals(req.CacheConfig) {
		return false
	}
	return prefix != "" && prefix == m.Fingerprint
}

// finish deletes the cache plan leaves behind, which depends on whether the
// response carrying its metadata reached the caller. It runs after the
// request has ended, so it uses a context that outlives a cancellation.
func (c *contextCache) finish(ctx context.Context, plan cachePlan, delivered bool) {
	name := plan.created
	if delivered {
		name = plan.stale
	}
	if name == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), cacheCleanupTimeout)
	defer cancel()
	// A failure is ignored: the cache expires on its own when its TTL runs
	// out.
	_, _ = c.client.Caches.Delete(ctx, name, &genai.DeleteCachedContentConfig{HTTPOptions: c.httpOptions(nil)})
}

// httpOptions returns a copy of base, or new options when base is nil, with
// ADK's tracking headers added. A copy, because genai writes into the options
// it is given and base is shared by every request of the runner.
func (c *contextCache) httpOptions(base *genai.HTTPOptions) *genai.HTTPOptions {
	out := &genai.HTTPOptions{}
	if base != nil {
		*out = *base
	}
	out.Headers = out.Headers.Clone()
	if out.Headers == nil {
		out.Headers = http.Header{}
	}
	out.Headers.Set("x-goog-api-client", c.trackingHeader)
	out.Headers.Set("user-agent", c.trackingHeader)
	return out
}

// countContentsToCache returns how many leading contents a cache holds:
// everything before the trailing run of user contents.
func countContentsToCache(contents []*genai.Content) int {
	n := len(contents)
	for n > 0 && contents[n-1] != nil && contents[n-1].Role == genai.RoleUser {
		n--
	}
	return n
}

// cacheScope names the backend that owns explicit caches, since a cache made
// under one project or endpoint cannot serve a request sent to another.
type cacheScope struct {
	Backend  string `json:"backend"`
	Project  string `json:"project,omitempty"`
	Location string `json:"location,omitempty"`
	BaseURL  string `json:"baseUrl,omitempty"`
}

func scopeFor(cfg genai.ClientConfig) cacheScope {
	s := cacheScope{Backend: "gemini", BaseURL: cfg.HTTPOptions.BaseURL}
	if cfg.Backend == genai.BackendVertexAI {
		s.Backend = "vertex"
		s.Project = cfg.Project
		s.Location = cfg.Location
	}
	return s
}

// fingerprintData is what a fingerprint hashes. Its field order fixes the
// encoding, and encoding/json sorts map keys, so equal prefixes encode alike.
type fingerprintData struct {
	Model             string            `json:"model"`
	Scope             cacheScope        `json:"cacheScope"`
	SystemInstruction *genai.Content    `json:"systemInstruction,omitempty"`
	Tools             []json.RawMessage `json:"tools,omitempty"`
	ToolConfig        *genai.ToolConfig `json:"toolConfig,omitempty"`
	Contents          []*genai.Content  `json:"cachedContents,omitempty"`
}

// fingerprinter fingerprints prefixes of one request. The tools are encoded
// once, since a request is fingerprinted at up to two lengths and a large
// toolset is most of the cost.
type fingerprinter struct {
	req  *model.LLMRequest
	data fingerprintData
	// ok is false when the request cannot be encoded.
	ok bool
}

func newFingerprinter(req *model.LLMRequest, modelName string, scope cacheScope) *fingerprinter {
	tools, err := canonicalTools(req.Config.Tools)
	return &fingerprinter{
		req: req,
		data: fingerprintData{
			Model:             modelName,
			Scope:             scope,
			SystemInstruction: req.Config.SystemInstruction,
			Tools:             tools,
			ToolConfig:        req.Config.ToolConfig,
		},
		ok: err == nil,
	}
}

// of returns a short hash of the model, the backend, the system instruction,
// the tools and their config, and the first count contents. A reordered tool
// list, or reordered function declarations within a tool, fingerprints the
// same. It returns "" when the request cannot be encoded, which matches no
// fingerprint and so keeps the request uncached.
func (f *fingerprinter) of(count int) string {
	if !f.ok {
		return ""
	}
	data := f.data
	if count > 0 {
		data.Contents = f.req.Contents[:min(count, len(f.req.Contents))]
	}
	encoded, err := json.Marshal(data)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:])[:16]
}

// metadataFor returns fingerprint-only metadata for the first count contents.
func (f *fingerprinter) metadataFor(count int) *model.CacheMetadata {
	return &model.CacheMetadata{Fingerprint: f.of(count), ContentsCount: count}
}

// canonicalTools encodes tools with each tool's function declarations sorted
// by name and the tools sorted by their encoding.
func canonicalTools(tools []*genai.Tool) ([]json.RawMessage, error) {
	out := make([]json.RawMessage, 0, len(tools))
	for _, t := range tools {
		if t == nil {
			continue
		}
		sorted := *t
		sorted.FunctionDeclarations = slices.SortedStableFunc(slices.Values(t.FunctionDeclarations), func(a, b *genai.FunctionDeclaration) int {
			return cmp.Compare(declarationName(a), declarationName(b))
		})
		encoded, err := json.Marshal(&sorted)
		if err != nil {
			return nil, err
		}
		out = append(out, encoded)
	}
	slices.SortFunc(out, func(a, b json.RawMessage) int { return strings.Compare(string(a), string(b)) })
	return out, nil
}

func declarationName(d *genai.FunctionDeclaration) string {
	if d == nil {
		return ""
	}
	return d.Name
}

// createCache creates a cache holding the first count contents of req, along
// with its system instruction, tools and tool config, and returns its metadata
// with fingerprint as the prefix's. It returns nil, creating nothing, when no
// earlier prompt size is known, when that prompt was below MinTokens, when the
// prefix is estimated to fall below the model's minimum cache size, or when
// the provider refuses.
func (c *contextCache) createCache(ctx context.Context, req *model.LLMRequest, count int, fingerprint string) *model.CacheMetadata {
	if req.PreviousPromptTokenCount <= 0 || fingerprint == "" {
		return nil
	}
	cfg := req.CacheConfig
	if req.PreviousPromptTokenCount < cfg.MinTokens {
		return nil
	}
	// The earlier prompt's count covers every content, but only the prefix
	// is cached. On a long conversation that count clears the model's
	// minimum while the prefix does not, and the provider then rejects the
	// cache.
	if floor := minCacheTokens(c.model); floor > 0 && estimatePrefixTokens(req, count) < floor {
		return nil
	}

	ttl := cacheTTL(cfg)
	createCfg := &genai.CreateCachedContentConfig{
		HTTPOptions:       c.httpOptions(cfg.CreateHTTPOptions),
		TTL:               ttl,
		DisplayName:       fmt.Sprintf("adk-cache-%d-%dcontents", platform.Now(ctx).Unix(), count),
		SystemInstruction: req.Config.SystemInstruction,
		Tools:             req.Config.Tools,
		ToolConfig:        req.Config.ToolConfig,
	}
	if count > 0 {
		createCfg.Contents = req.Contents[:count]
	}
	cached, err := c.client.Caches.Create(ctx, c.model, createCfg)
	if err != nil || cached == nil || cached.Name == "" {
		return nil
	}

	createdAt := platform.Now(ctx)
	expireTime := cached.ExpireTime
	if expireTime.IsZero() {
		expireTime = createdAt.Add(ttl)
	}
	return &model.CacheMetadata{
		CacheName:       cached.Name,
		ExpireTime:      expireTime,
		InvocationsUsed: 1,
		CreatedAt:       createdAt,
		Fingerprint:     fingerprint,
		ContentsCount:   count,
	}
}

// applyCache returns a copy of req that uses the named cache, which holds the
// first count contents. The copy leaves out the system instruction, tools and
// tool config, which the provider rejects alongside a cache, and the cached
// contents. It always keeps the last content, because a request with no
// contents is rejected.
func applyCache(req *model.LLMRequest, name string, count int) *model.LLMRequest {
	out := *req
	cfg := *req.Config
	cfg.SystemInstruction = nil
	cfg.Tools = nil
	cfg.ToolConfig = nil
	cfg.CachedContent = name
	out.Config = &cfg
	out.Contents = req.Contents[min(count, max(len(req.Contents)-1, 0)):]
	return &out
}

// minCacheTokens returns the smallest explicit cache the named model accepts,
// or 0 when its name does not say.
func minCacheTokens(modelName string) int32 {
	name := modelName[strings.LastIndex(modelName, "/")+1:]
	switch {
	case strings.HasPrefix(name, "gemini-2.5-"):
		return gemini25MinCacheTokens
	case strings.HasPrefix(name, "gemini-3"):
		return gemini3MinCacheTokens
	default:
		return 0
	}
}

// estimatePrefixTokens estimates the size of what a cache of the first count
// contents would hold. The only real count is PreviousPromptTokenCount, which
// covers the whole earlier prompt, so it is scaled by the prefix's share of
// the request's characters.
//
// adk-python scales by the same share but counts tools by their encoding with
// null fields included, which weighs them more. The two can disagree about a
// prefix within a few percent of the minimum, and the provider has the final
// word either way.
func estimatePrefixTokens(req *model.LLMRequest, count int) int32 {
	full := req.PreviousPromptTokenCount
	fixed := textChars(req.Config.SystemInstruction)
	for _, t := range req.Config.Tools {
		if encoded, err := json.Marshal(t); err == nil {
			fixed += len(encoded)
		}
	}
	prefix, total := fixed, fixed
	for i, content := range req.Contents {
		chars := textChars(content)
		total += chars
		if i < count {
			prefix += chars
		}
	}
	if total == 0 {
		// Nothing to measure the share by, for example only non-text parts.
		// The real count is a better guess than skipping the cache.
		return full
	}
	return int32(float64(full) * float64(prefix) / float64(total))
}

func textChars(content *genai.Content) int {
	if content == nil {
		return 0
	}
	chars := 0
	for _, part := range content.Parts {
		if part != nil {
			chars += len(part.Text)
		}
	}
	return chars
}

func cacheIntervals(cfg *model.ContextCacheConfig) int {
	if cfg.CacheIntervals == 0 {
		return defaultCacheIntervals
	}
	return cfg.CacheIntervals
}

func cacheTTL(cfg *model.ContextCacheConfig) time.Duration {
	if cfg.TTL == 0 {
		return defaultCacheTTL
	}
	return cfg.TTL
}
