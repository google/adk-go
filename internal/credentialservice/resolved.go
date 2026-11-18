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

package credentialservice

import (
	"context"
	"sync"

	"google.golang.org/adk/v2/tool/toolauth"
)

// Resolved holds the credentials toolset authentication resolved in one run,
// by credential key. It is adk-python's InvocationContext.credential_by_key.
// A nil *Resolved holds nothing and drops every Set. Safe for concurrent use.
type Resolved struct {
	mu    sync.Mutex
	creds map[string]*toolauth.Credential
}

type resolvedKey struct{}

// WithResolved returns a context derived from ctx that carries a new, empty
// Resolved, in place of any ctx carries.
func WithResolved(ctx context.Context) context.Context {
	return context.WithValue(ctx, resolvedKey{}, &Resolved{creds: map[string]*toolauth.Credential{}})
}

// ResolvedFrom returns the Resolved the closest WithResolved attached to ctx,
// or nil.
func ResolvedFrom(ctx context.Context) *Resolved {
	if ctx == nil {
		return nil
	}
	r, _ := ctx.Value(resolvedKey{}).(*Resolved)
	return r
}

// Set stores a copy of cred under key.
func (r *Resolved) Set(key string, cred *toolauth.Credential) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.creds[key] = cred.Clone()
}

// Get returns a copy of the credential under key, or nil.
func (r *Resolved) Get(key string) *toolauth.Credential {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.creds[key].Clone()
}
