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
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
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
func NewAuthConfig(scheme AuthScheme, credential *AuthCredential) (*AuthConfig, error) {
	cfg := &AuthConfig{
		AuthScheme:        scheme,
		RawAuthCredential: credential,
	}
	if cfg.CredentialKey == "" {
		key, err := cfg.generateCredentialKey()
		if err != nil {
			return nil, fmt.Errorf("generate credential key: %w", err)
		}
		cfg.CredentialKey = key
	}
	return cfg, nil
}

// generateCredentialKey creates a unique key based on auth scheme and credential.
func (c *AuthConfig) generateCredentialKey() (string, error) {
	var schemePart, credPart string
	if c.AuthScheme != nil {
		schemeJSON, err := stableJSON(c.AuthScheme)
		if err != nil {
			return "", fmt.Errorf("marshal auth scheme: %w", err)
		}
		schemeType := c.AuthScheme.GetType()
		h := sha256.Sum256([]byte(schemeJSON))
		schemePart = fmt.Sprintf("%s_%x", schemeType, h[:8])
	}
	if c.RawAuthCredential != nil {
		credJSON, err := stableJSON(c.RawAuthCredential)
		if err != nil {
			return "", fmt.Errorf("marshal auth credential: %w", err)
		}
		h := sha256.Sum256([]byte(credJSON))
		credPart = fmt.Sprintf("%s_%x", c.RawAuthCredential.AuthType, h[:8])
	}
	return fmt.Sprintf("adk_%s_%s", schemePart, credPart), nil
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

// stableJSON returns a deterministic JSON representation with sorted map keys.
func stableJSON(v interface{}) (string, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	var data interface{}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&data); err != nil {
		return "", err
	}
	var buf bytes.Buffer
	if err := encodeCanonical(&buf, data); err != nil {
		return "", err
	}
	return buf.String(), nil
}

func encodeCanonical(buf *bytes.Buffer, v interface{}) error {
	switch val := v.(type) {
	case nil:
		buf.WriteString("null")
	case bool:
		if val {
			buf.WriteString("true")
		} else {
			buf.WriteString("false")
		}
	case string:
		buf.WriteString(strconv.Quote(val))
	case json.Number:
		buf.WriteString(val.String())
	case float64:
		buf.WriteString(strconv.FormatFloat(val, 'g', -1, 64))
	case []interface{}:
		buf.WriteByte('[')
		for i, elem := range val {
			if i > 0 {
				buf.WriteByte(',')
			}
			if err := encodeCanonical(buf, elem); err != nil {
				return err
			}
		}
		buf.WriteByte(']')
	case map[string]interface{}:
		buf.WriteByte('{')
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for i, k := range keys {
			if i > 0 {
				buf.WriteByte(',')
			}
			buf.WriteString(strconv.Quote(k))
			buf.WriteByte(':')
			if err := encodeCanonical(buf, val[k]); err != nil {
				return err
			}
		}
		buf.WriteByte('}')
	default:
		return fmt.Errorf("unsupported JSON canonicalization type %T", v)
	}
	return nil
}
