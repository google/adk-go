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

package a2aclient

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/log"
)

// ErrCredentialNotFound is returned by [CredentialsService] if a credential for the provided
// (sessionId, scheme) pair was not found.
var ErrCredentialNotFound = errors.New("credential not found")

// SessionID is a client-generated identifier used for scoping auth credentials.
type SessionID string

// Used to store a SessionID in context.Context.
type sessionIDKey struct{}

// WithSessionID allows callers to attach a session identifier to the request.
// [CallInterceptor] can access this identifier using [SessionIDFrom].
func WithSessionID(ctx context.Context, sid SessionID) context.Context {
	return context.WithValue(ctx, sessionIDKey{}, sid)
}

// SessionIDFrom allows to get a previously attached session identifier from Context.
func SessionIDFrom(ctx context.Context) (SessionID, bool) {
	sid, ok := ctx.Value(sessionIDKey{}).(SessionID)
	return sid, ok
}

// AuthCredential represents a security-scheme specific credential (eg. a JWT token).
type AuthCredential string

// AuthInterceptor implements [CallInterceptor].
// It uses SessionID provided using [WithSessionID] to lookup credentials
// and attach them according to the security scheme specified in the agent card.
// Credentials fetching is delegated to [CredentialsService].
type AuthInterceptor struct {
	PassthroughInterceptor
	Service CredentialsService
}

func (ai *AuthInterceptor) Before(ctx context.Context, req *Request) (context.Context, error) {
	if req.Card == nil || req.Card.Security == nil || req.Card.SecuritySchemes == nil {
		return ctx, nil
	}

	sessionID, ok := SessionIDFrom(ctx)
	if !ok {
		return ctx, nil
	}

	for _, requirement := range req.Card.Security {
		for schemeName := range requirement {
			credential, err := ai.Service.Get(ctx, sessionID, schemeName)
			if errors.Is(err, ErrCredentialNotFound) {
				continue
			}
			if err != nil {
				log.Error(ctx, "credentials service error", err)
				continue
			}
			scheme, ok := req.Card.SecuritySchemes[schemeName]
			if !ok {
				continue
			}
			switch v := scheme.(type) {
			case a2a.HTTPAuthSecurityScheme, a2a.OAuth2SecurityScheme:
				req.Meta["Authorization"] = []string{fmt.Sprintf("Bearer %s", credential)}
				return ctx, nil
			case a2a.APIKeySecurityScheme:
				req.Meta[v.Name] = []string{string(credential)}
				return ctx, nil
			}
		}
	}

	return ctx, nil
}

// CredentialsService is used by [AuthInterceptor] for resolving credentials.
type CredentialsService interface {
	Get(ctx context.Context, sid SessionID, scheme a2a.SecuritySchemeName) (AuthCredential, error)
}

// SessionCredentials is a map of scheme names to auth credentials.
type SessionCredentials map[a2a.SecuritySchemeName]AuthCredential

// InMemoryCredentialsStore implements [CredentialsService].
type InMemoryCredentialsStore struct {
	mu          sync.RWMutex
	credentials map[SessionID]SessionCredentials
}

var _ CredentialsService = (*InMemoryCredentialsStore)(nil)

// NewInMemoryCredentialsStore initializes an in-memory implementation of [CredentialsService].
func NewInMemoryCredentialsStore() *InMemoryCredentialsStore {
	return &InMemoryCredentialsStore{
		credentials: make(map[SessionID]SessionCredentials),
	}
}

func (s *InMemoryCredentialsStore) Get(ctx context.Context, sid SessionID, scheme a2a.SecuritySchemeName) (AuthCredential, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	forSession, ok := s.credentials[sid]
	if !ok {
		return AuthCredential(""), ErrCredentialNotFound
	}

	credential, ok := forSession[scheme]
	if !ok {
		return AuthCredential(""), ErrCredentialNotFound
	}

	return credential, nil
}

func (s *InMemoryCredentialsStore) Set(sid SessionID, scheme a2a.SecuritySchemeName, credential AuthCredential) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.credentials[sid]; !ok {
		s.credentials[sid] = make(map[a2a.SecuritySchemeName]AuthCredential)
	}
	s.credentials[sid][scheme] = credential
}
