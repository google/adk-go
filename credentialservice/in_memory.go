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
	"errors"
	"fmt"
	"sync"

	"google.golang.org/adk/v2/tool/toolauth"
)

// InMemoryService returns a [Service] that keeps credentials in memory, for
// tests and for programs that need no credential to outlive the process. Each
// call returns a new, empty service. Runners that share one service share its
// credentials, still kept apart by app name and user ID.
//
// The service stores a copy of each saved credential and returns a copy on
// each load ([toolauth.Credential.Clone]). The returned service is safe for
// concurrent use.
func InMemoryService() Service {
	return &inMemoryService{credentials: make(map[storeKey]*toolauth.Credential)}
}

// storeKey addresses one stored credential. A struct key keeps the three
// parts apart, so ("ab", "c") and ("a", "bc") never meet.
type storeKey struct {
	appName, userID, credentialKey string
}

type inMemoryService struct {
	mu sync.RWMutex // guards credentials
	// credentials holds copies the service owns; nobody outside it sees them.
	credentials map[storeKey]*toolauth.Credential
}

var errNilConfig = errors.New("credentialservice: auth config is nil")

func (s *inMemoryService) LoadCredential(ctx context.Context, appName, userID string, config *toolauth.Config) (*toolauth.Credential, error) {
	k, err := newStoreKey(appName, userID, config)
	if err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.credentials[k].Clone(), nil
}

func (s *inMemoryService) SaveCredential(ctx context.Context, appName, userID string, config *toolauth.Config) error {
	k, err := newStoreKey(appName, userID, config)
	if err != nil {
		return err
	}
	cred := config.ExchangedAuthCredential.Clone()
	s.mu.Lock()
	defer s.mu.Unlock()
	if cred == nil {
		// adk-python stores None, which a later load returns as no credential.
		delete(s.credentials, k)
		return nil
	}
	s.credentials[k] = cred
	return nil
}

func newStoreKey(appName, userID string, config *toolauth.Config) (storeKey, error) {
	if config == nil {
		return storeKey{}, errNilConfig
	}
	key, err := config.Key()
	if err != nil {
		return storeKey{}, fmt.Errorf("credentialservice: %w", err)
	}
	return storeKey{appName: appName, userID: userID, credentialKey: key}, nil
}
