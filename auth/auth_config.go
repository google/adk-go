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
	"crypto/sha256"
	"encoding/json"
	"fmt"
)

// AuthConfig combines auth scheme and credentials for a tool.
// This is passed to tools that require authentication.
type AuthConfig struct {
	// AuthScheme defines how the API expects authentication.
	AuthScheme AuthScheme `json:"authScheme"`
	// RawAuthCredential is the initial credential (e.g., client_id/secret).
	RawAuthCredential *AuthCredential `json:"rawAuthCredential,omitempty"`
	// ExchangedAuthCredential is the processed credential (e.g., access_token).
	ExchangedAuthCredential *AuthCredential `json:"exchangedAuthCredential,omitempty"`
	// CredentialKey is a unique key for persisting this credential.
	CredentialKey string `json:"credentialKey,omitempty"`
}

// NewAuthConfig creates a new AuthConfig with the given scheme and credential.
// If credentialKey is empty, it will be generated automatically.
func NewAuthConfig(scheme AuthScheme, credential *AuthCredential) *AuthConfig {
	cfg := &AuthConfig{
		AuthScheme:        scheme,
		RawAuthCredential: credential,
	}
	if cfg.CredentialKey == "" {
		cfg.CredentialKey = cfg.generateCredentialKey()
	}
	return cfg
}

// generateCredentialKey creates a unique key based on auth scheme and credential.
func (c *AuthConfig) generateCredentialKey() string {
	var schemePart string
	if c.AuthScheme != nil {
		schemeType := c.AuthScheme.GetType()
		schemeJSON, _ := json.Marshal(c.AuthScheme)
		h := sha256.Sum256(schemeJSON)
		schemePart = fmt.Sprintf("%s_%x", schemeType, h[:8])
	}

	var credPart string
	if c.RawAuthCredential != nil {
		credJSON, _ := json.Marshal(c.RawAuthCredential)
		h := sha256.Sum256(credJSON)
		credPart = fmt.Sprintf("%s_%x", c.RawAuthCredential.AuthType, h[:8])
	}

	return fmt.Sprintf("adk_%s_%s", schemePart, credPart)
}

// Copy creates a deep copy of the AuthConfig.
func (c *AuthConfig) Copy() *AuthConfig {
	if c == nil {
		return nil
	}
	return &AuthConfig{
		AuthScheme:              c.AuthScheme, // AuthScheme is typically immutable
		RawAuthCredential:       c.RawAuthCredential.Copy(),
		ExchangedAuthCredential: c.ExchangedAuthCredential.Copy(),
		CredentialKey:           c.CredentialKey,
	}
}
