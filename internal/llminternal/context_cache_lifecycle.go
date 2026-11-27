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

package llminternal

import (
	"context"
	"maps"
	"reflect"
	"slices"
	"sync"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// ContextCacheLedger collects the context caches that one invocation's model
// calls created or stopped using, so that which of them to delete is decided
// once the invocation's events are stored rather than when the model returns.
//
// The model cannot decide it. Between its response and the stored event sit
// after-model callbacks, plugins, the flow's own filtering and the session
// write, and any of them can store a different record than the model
// returned, or none. Deleting on the model's guess deletes caches the session
// still names and leaks caches it never recorded.
//
// [ContextCacheLedger.Settle] applies one rule against the stored events: keep
// every cache that an agent's latest stored record names, and delete every
// other cache the ledger holds. So a cache is deleted only when no agent's
// latest record names it, and every cache a model call created is either named
// by one or deleted. Only the latest record counts because the request
// processor reads only that: a cache an older record names can no longer be
// used.
//
// It is safe for concurrent use, since parallel sub-agents share an
// invocation.
type ContextCacheLedger struct {
	mu sync.Mutex
	// pending maps each cache's name to the call that deletes it.
	pending map[string]func(context.Context)
}

// NewContextCacheLedger returns an empty ledger for one invocation.
func NewContextCacheLedger() *ContextCacheLedger {
	return &ContextCacheLedger{pending: make(map[string]func(context.Context))}
}

// Track adds the named cache to those Settle decides on, with the call that
// deletes it. Tracking a name again replaces its delete call.
func (l *ContextCacheLedger) Track(name string, deleteCache func(context.Context)) {
	if l == nil || name == "" || deleteCache == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.pending[name] = deleteCache
}

// Settle deletes every tracked cache that no agent's latest stored record
// names, and empties the ledger. Deletions run in name order, so the outcome
// does not depend on map iteration.
//
// load returns the events as the session store holds them. It is called only
// when something is tracked, so an invocation that merely reused a cache reads
// nothing. If it fails, Settle deletes nothing: without the stored records it
// cannot tell which caches are named, and a cache left on the provider costs
// storage until its TTL, while a cache deleted under a record that names it
// fails the next request that uses it.
func (l *ContextCacheLedger) Settle(ctx context.Context, load func(context.Context) (session.Events, error)) {
	if l == nil {
		return
	}
	l.mu.Lock()
	pending := l.pending
	l.pending = make(map[string]func(context.Context))
	l.mu.Unlock()
	if len(pending) == 0 {
		return
	}

	events, err := load(ctx)
	if err != nil {
		return
	}
	named := latestCacheNames(events)
	for _, name := range slices.Sorted(maps.Keys(pending)) {
		if !named[name] {
			pending[name](ctx)
		}
	}
}

type contextCacheLedgerKey struct{}

// WithContextCacheLedger returns a copy of ctx carrying l, which the model
// tracks the caches of its calls in. A nil l means nothing settles them, and
// the model then deletes caches on its own as the response reaches the
// caller, which is the best it can do without a runner.
func WithContextCacheLedger(ctx context.Context, l *ContextCacheLedger) context.Context {
	return context.WithValue(ctx, contextCacheLedgerKey{}, l)
}

// ContextCacheLedgerFromContext returns the ledger [WithContextCacheLedger]
// put on ctx, or nil when there is none.
func ContextCacheLedgerFromContext(ctx context.Context) *ContextCacheLedger {
	l, _ := ctx.Value(contextCacheLedgerKey{}).(*ContextCacheLedger)
	return l
}

// latestCacheNames returns the caches that the latest cache record of each
// agent names, where an agent is one cacheScope. A scope whose latest record
// is fingerprint-only names none, even when an older record named a cache:
// the processor reads only the latest, so that cache can no longer be used.
func latestCacheNames(events session.Events) map[string]bool {
	named := make(map[string]bool)
	if events == nil {
		return named
	}
	seen := make(map[cacheScope]bool)
	for i := events.Len() - 1; i >= 0; i-- {
		ev := events.At(i)
		if ev == nil || ev.CacheMetadata == nil {
			continue
		}
		scope := cacheScope{agentName: ev.Author, branch: ev.Branch, isolationScope: ev.IsolationScope}
		if seen[scope] {
			continue
		}
		seen[scope] = true
		if ev.CacheMetadata.Active() {
			named[ev.CacheMetadata.CacheName] = true
		}
	}
	return named
}

// CloneContextCacheConfig returns a copy of cfg whose CreateHTTPOptions it
// shares nothing mutable with, as [cloneHTTPOptions] describes, or nil for a
// nil cfg.
//
// CreateHTTPOptions holds maps and pointers, and genai writes into the
// options it is given. A copy one level deep would leave every request
// sharing the caller's header map, so a caller or a callback writing to it
// would race with the requests reading it.
func CloneContextCacheConfig(cfg *model.ContextCacheConfig) *model.ContextCacheConfig {
	if cfg == nil {
		return nil
	}
	out := *cfg
	out.CreateHTTPOptions = cloneHTTPOptions(cfg.CreateHTTPOptions)
	return &out
}

// cloneHTTPOptions copies o so that writing to the copy's headers, timeout,
// retry options, or any map or slice in its extra body leaves o unchanged.
//
// ExtraBody is copied through its maps, slices and interface values, so a
// callback that writes to a map nested in it, such as a labels map, writes to
// the copy. A value of any other kind is copied the way Go copies it: a struct
// by value, so a map inside a struct is still shared, and a pointer as the
// pointer. ExtrasRequestProvider is a function and is shared as it is.
func cloneHTTPOptions(o *genai.HTTPOptions) *genai.HTTPOptions {
	if o == nil {
		return nil
	}
	out := *o
	out.Headers = o.Headers.Clone()
	out.ExtraBody = cloneBody(o.ExtraBody)
	if o.Timeout != nil {
		timeout := *o.Timeout
		out.Timeout = &timeout
	}
	if o.RetryOptions != nil {
		retry := *o.RetryOptions
		retry.Attempts = clonePointer(o.RetryOptions.Attempts)
		retry.InitialDelay = clonePointer(o.RetryOptions.InitialDelay)
		retry.MaxDelay = clonePointer(o.RetryOptions.MaxDelay)
		retry.ExpBase = clonePointer(o.RetryOptions.ExpBase)
		retry.Jitter = clonePointer(o.RetryOptions.Jitter)
		retry.HTTPStatusCodes = slices.Clone(o.RetryOptions.HTTPStatusCodes)
		out.RetryOptions = &retry
	}
	return &out
}

func clonePointer[T any](p *T) *T {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

// cloneBody returns a copy of m, which keeps its nil-ness, through
// [cloneValue].
func cloneBody(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	return cloneValue(reflect.ValueOf(m)).Interface().(map[string]any)
}

// cloneValue copies maps, slices and interface values recursively, and returns
// a value of any other kind as it is, which for a struct is a copy of its
// fields and for a pointer is the pointer. A nil map or slice stays nil.
//
// It does not look inside structs or follow pointers, so it needs no case for
// unexported fields, and a value that refers to itself only through them
// cannot loop. A map or slice that contains itself would, but such a value
// cannot be sent as a JSON body either.
func cloneValue(v reflect.Value) reflect.Value {
	switch v.Kind() {
	case reflect.Map:
		if v.IsNil() {
			return v
		}
		out := reflect.MakeMapWithSize(v.Type(), v.Len())
		for iter := v.MapRange(); iter.Next(); {
			out.SetMapIndex(iter.Key(), cloneValue(iter.Value()))
		}
		return out
	case reflect.Slice:
		if v.IsNil() {
			return v
		}
		out := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		for i := range v.Len() {
			out.Index(i).Set(cloneValue(v.Index(i)))
		}
		return out
	case reflect.Interface:
		if v.IsNil() {
			return v
		}
		out := reflect.New(v.Type()).Elem()
		out.Set(cloneValue(v.Elem()))
		return out
	default:
		return v
	}
}
