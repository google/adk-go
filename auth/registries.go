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

// ExchangeResult contains the result of a credential exchange.
type ExchangeResult struct {
	// Credential is the exchanged credential.
	Credential *AuthCredential
	// WasExchanged indicates if the credential was actually exchanged.
	WasExchanged bool
}

// CredentialExchanger exchanges credentials from one form to another.
// For example, exchanging an authorization code for an access token.
type CredentialExchanger interface {
	// Exchange exchanges the given credential using the auth scheme.
	// Returns the exchanged credential and whether it was exchanged.
	Exchange(ctx context.Context, cred *AuthCredential, scheme AuthScheme) (*ExchangeResult, error)
}

// CredentialRefresher refreshes expired credentials.
type CredentialRefresher interface {
	// IsRefreshNeeded checks if the credential needs to be refreshed.
	IsRefreshNeeded(cred *AuthCredential, scheme AuthScheme) bool

	// Refresh refreshes the credential and returns the new credential.
	Refresh(ctx context.Context, cred *AuthCredential, scheme AuthScheme) (*AuthCredential, error)
}

// ExchangerRegistry manages credential exchangers by credential type.
type ExchangerRegistry struct {
	mu         sync.RWMutex
	exchangers map[AuthCredentialType]CredentialExchanger
}

// NewExchangerRegistry creates a new exchanger registry.
func NewExchangerRegistry() *ExchangerRegistry {
	return &ExchangerRegistry{
		exchangers: make(map[AuthCredentialType]CredentialExchanger),
	}
}

// Register registers an exchanger for a credential type.
func (r *ExchangerRegistry) Register(credType AuthCredentialType, exchanger CredentialExchanger) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.exchangers[credType] = exchanger
}

// Get returns the exchanger for a credential type, or nil if not found.
func (r *ExchangerRegistry) Get(credType AuthCredentialType) CredentialExchanger {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.exchangers[credType]
}

// RefresherRegistry manages credential refreshers by credential type.
type RefresherRegistry struct {
	mu         sync.RWMutex
	refreshers map[AuthCredentialType]CredentialRefresher
}

// NewRefresherRegistry creates a new refresher registry.
func NewRefresherRegistry() *RefresherRegistry {
	return &RefresherRegistry{
		refreshers: make(map[AuthCredentialType]CredentialRefresher),
	}
}

// Register registers a refresher for a credential type.
func (r *RefresherRegistry) Register(credType AuthCredentialType, refresher CredentialRefresher) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.refreshers[credType] = refresher
}

// Get returns the refresher for a credential type, or nil if not found.
func (r *RefresherRegistry) Get(credType AuthCredentialType) CredentialRefresher {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.refreshers[credType]
}
