// Copyright 2025 Google LLC
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

package auth

import (
	"context"
	"sync"
)

// CredentialService is an interface for loading and saving credentials.
type CredentialService interface {
	// LoadCredential loads a credential from storage.
	LoadCredential(ctx context.Context, config *AuthConfig) (*AuthCredential, error)

	// SaveCredential saves a credential to storage.
	SaveCredential(ctx context.Context, config *AuthConfig) error
}

// InMemoryCredentialService stores credentials in memory.
type InMemoryCredentialService struct {
	mu          sync.RWMutex
	credentials map[string]*AuthCredential
}

// NewInMemoryCredentialService creates a new in-memory credential service.
func NewInMemoryCredentialService() *InMemoryCredentialService {
	return &InMemoryCredentialService{
		credentials: make(map[string]*AuthCredential),
	}
}

// LoadCredential loads a credential from memory.
func (s *InMemoryCredentialService) LoadCredential(ctx context.Context, config *AuthConfig) (*AuthCredential, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if cred, ok := s.credentials[config.CredentialKey]; ok {
		return cred.Copy(), nil
	}
	return nil, nil
}

// SaveCredential saves a credential to memory.
func (s *InMemoryCredentialService) SaveCredential(ctx context.Context, config *AuthConfig) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if config.ExchangedAuthCredential != nil {
		s.credentials[config.CredentialKey] = config.ExchangedAuthCredential.Copy()
	}
	return nil
}

// SessionStateCredentialService stores credentials in session state.
type SessionStateCredentialService struct {
	stateSetter func(key string, value interface{})
	stateGetter func(key string) interface{}
}

// NewSessionStateCredentialService creates a new session state credential service.
func NewSessionStateCredentialService(
	getter func(key string) interface{},
	setter func(key string, value interface{}),
) *SessionStateCredentialService {
	return &SessionStateCredentialService{
		stateGetter: getter,
		stateSetter: setter,
	}
}

// LoadCredential loads a credential from session state.
func (s *SessionStateCredentialService) LoadCredential(ctx context.Context, config *AuthConfig) (*AuthCredential, error) {
	key := "cred:" + config.CredentialKey
	if val := s.stateGetter(key); val != nil {
		if cred, ok := val.(*AuthCredential); ok {
			return cred.Copy(), nil
		}
	}
	return nil, nil
}

// SaveCredential saves a credential to session state.
func (s *SessionStateCredentialService) SaveCredential(ctx context.Context, config *AuthConfig) error {
	if config.ExchangedAuthCredential != nil {
		key := "cred:" + config.CredentialKey
		s.stateSetter(key, config.ExchangedAuthCredential.Copy())
	}
	return nil
}
