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

package models

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"sort"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/tool/toolauth"
)

// credentialSecretKeys are the JSON keys of credential fields that can hold a
// secret. Both spellings are listed because Extra keeps a key under the name
// it arrived with.
var credentialSecretKeys = map[string]bool{
	"password":           true,
	"token":              true,
	"additionalHeaders":  true,
	"additional_headers": true,
	"clientSecret":       true,
	"client_secret":      true,
	"authResponseUri":    true,
	"auth_response_uri":  true,
	"authCode":           true,
	"auth_code":          true,
	"accessToken":        true,
	"access_token":       true,
	"refreshToken":       true,
	"refresh_token":      true,
	"idToken":            true,
	"id_token":           true,
	"codeVerifier":       true,
	"code_verifier":      true,
	"privateKeyId":       true,
	"private_key_id":     true,
	"privateKey":         true,
	"private_key":        true,
	"apiKey":             true,
	"api_key":            true,
}

// authConfigs is a session's requested auth configs, keyed by function call
// ID. Its encoding leaves out credentialSecretKeys at any depth, so every
// response that carries an event drops them.
type authConfigs map[string]toolauth.Config

// MarshalJSON implements [encoding/json.Marshaler].
func (c authConfigs) MarshalJSON() ([]byte, error) {
	m, err := withoutSecretKeys(map[string]toolauth.Config(c))
	if err != nil {
		return nil, fmt.Errorf("requested auth configs: %w", err)
	}
	return json.Marshal(m)
}

// withoutCredentialSecrets returns p, or a copy of it when p is part of an
// adk_request_credential call or answer, with credentialSecretKeys left out of
// the args or the response. It never changes p: ADK reads the PKCE verifier
// and the authorization code from the stored event later.
func withoutCredentialSecrets(p *genai.Part) (*genai.Part, error) {
	out := *p
	switch fc, fr := p.FunctionCall, p.FunctionResponse; {
	case fc != nil && fc.Name == toolauth.RequestCredentialFunctionCallName:
		args, err := withoutSecretKeys(fc.Args)
		if err != nil {
			return nil, fmt.Errorf("%s args: %w", fc.Name, err)
		}
		call := *fc
		call.Args = args
		out.FunctionCall = &call
	case fr != nil && fr.Name == toolauth.RequestCredentialFunctionCallName:
		resp, err := withoutSecretKeys(fr.Response)
		if err != nil {
			return nil, fmt.Errorf("%s response: %w", fr.Name, err)
		}
		answer := *fr
		answer.Response = resp
		out.FunctionResponse = &answer
	default:
		return p, nil
	}
	return &out, nil
}

// stateValues is a session's state or an event's state delta. Its encoding
// leaves the secrets out of every credential and auth config in it, such as the
// tokens a tool's sign-in keeps in state, and keeps every other value whole.
type stateValues map[string]any

// MarshalJSON implements [encoding/json.Marshaler].
func (v stateValues) MarshalJSON() ([]byte, error) {
	m, err := asJSONObject(map[string]any(v))
	if err != nil {
		return nil, fmt.Errorf("state: %w", err)
	}
	dropCredentialSecrets(m)
	return json.Marshal(m)
}

// withoutSecretKeys returns v as a generic JSON object with
// credentialSecretKeys left out at any depth.
func withoutSecretKeys(v any) (map[string]any, error) {
	m, err := asJSONObject(v)
	if err != nil {
		return nil, err
	}
	dropSecretKeys(m)
	return m, nil
}

// asJSONObject encodes v, which must encode to a JSON object or null, and
// decodes it back into generic values. Going through JSON lets a struct value
// be cleaned the same way as a map.
func asJSONObject(v any) (map[string]any, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	// A number goes out as it was written, not through float64.
	dec.UseNumber()
	var m map[string]any
	if err := dec.Decode(&m); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return m, nil
}

// credentialTypeNames are the authType values that mark a JSON object as a
// credential: the values, names and lower-case names of adk-python's
// AuthCredentialTypes.
var credentialTypeNames = map[string]bool{
	"apiKey": true, "http": true, "oauth2": true, "openIdConnect": true, "serviceAccount": true,
	"API_KEY": true, "HTTP": true, "OAUTH2": true, "OPEN_ID_CONNECT": true, "SERVICE_ACCOUNT": true,
	"api_key": true, "open_id_connect": true, "service_account": true,
}

// credentialFields are the auth config keys whose values are credentials.
var credentialFields = map[string]bool{
	"rawAuthCredential":         true,
	"raw_auth_credential":       true,
	"exchangedAuthCredential":   true,
	"exchanged_auth_credential": true,
}

// dropCredentialSecrets deletes credentialSecretKeys from every credential in
// v, wherever it sits, finding them as adk-python's _redact_credential_secrets
// does: an object whose authType names a credential type, and the values of an
// auth config's credential fields.
func dropCredentialSecrets(v any) {
	switch v := v.(type) {
	case map[string]any:
		if isCredential(v) {
			dropSecretKeys(v)
			return
		}
		for k, val := range v {
			if credentialFields[k] {
				dropSecretKeys(val)
			} else {
				dropCredentialSecrets(val)
			}
		}
	case []any:
		for _, item := range v {
			dropCredentialSecrets(item)
		}
	}
}

func isCredential(m map[string]any) bool {
	for _, k := range []string{"authType", "auth_type"} {
		if name, ok := m[k].(string); ok && credentialTypeNames[name] {
			return true
		}
	}
	return false
}

// dropSecretKeys deletes credentialSecretKeys from every JSON object in v.
func dropSecretKeys(v any) {
	switch v := v.(type) {
	case map[string]any:
		maps.DeleteFunc(v, func(k string, _ any) bool { return credentialSecretKeys[k] })
		for _, val := range v {
			dropSecretKeys(val)
		}
	case []any:
		for _, item := range v {
			dropSecretKeys(item)
		}
	}
}

// MergeRedactedValue returns submitted with the credential secrets that a read
// strips from storedValue restored. A client that edits a field next to a
// credential sends the credential back as the read returned it, without the
// secrets; writing that value back unchanged would erase them.
//
// Only omitted credential secret keys are restored. submitted stays
// authoritative for every other field, including a secret the client did send.
// It reports whether submitted was changed.
func MergeRedactedValue(storedValue, submitted any) (any, bool, error) {
	stored, err := asJSONValue(storedValue)
	if err != nil {
		return submitted, false, err
	}
	merged, restored := restoreCredentialSecrets(stored, submitted, false)
	return merged, restored, nil
}

// asJSONValue encodes v and decodes it into generic JSON values, keeping
// numbers as json.Number so a stored value compares and round-trips exactly.
func asJSONValue(v any) (any, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var out any
	if err := dec.Decode(&out); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return out, nil
}

// restoreCredentialSecrets walks submitted next to the stored value and fills
// in the credential secret keys the submitted value omits. A key is treated as
// a secret once the walk is inside a credential, and for the values of an auth
// config's credential fields. A credential that carries any secret of its own
// is treated as an intentional replacement, so nothing is restored into it.
// Other omitted fields are left out, so the submitted value remains a full
// replacement of the value outside credentials.
func restoreCredentialSecrets(stored, submitted any, restoreSecrets bool) (any, bool) {
	switch s := submitted.(type) {
	case map[string]any:
		st, ok := stored.(map[string]any)
		if !ok {
			return submitted, false
		}
		if isCredential(st) {
			// Restore only into an object that outbound redaction still treats
			// as a credential. A client that clears or changes authType is
			// replacing the credential; restoring secrets into the ordinary
			// object would make the serializer leave them visible.
			restoreSecrets = isCredential(s) && !hasSecretKey(s)
		}
		restored := false
		if restoreSecrets {
			for k, v := range st {
				if _, ok := s[k]; !ok && credentialSecretKeys[k] {
					s[k] = v
					restored = true
				}
			}
		}
		for k, sub := range s {
			sv, ok := st[k]
			if !ok {
				continue
			}
			if merged, changed := restoreCredentialSecrets(sv, sub, restoreSecrets); changed {
				s[k] = merged
				restored = true
			}
		}
		return s, restored
	case []any:
		st, ok := stored.([]any)
		if !ok || len(st) != len(s) {
			return submitted, false
		}
		restored := false
		for i := range s {
			match := matchStoredElement(st, s[i])
			if match < 0 {
				continue
			}
			if merged, changed := restoreCredentialSecrets(st[match], s[i], restoreSecrets); changed {
				s[i] = merged
				restored = true
			}
		}
		return s, restored
	default:
		return submitted, false
	}
}

// matchStoredElement returns the index of the stored array element that
// identifies the same credential as submitted, or -1 when none does or more
// than one does. Matching by a stable non-secret identity keeps a reordered
// array from handing one entry another entry's secrets.
func matchStoredElement(stored []any, submitted any) int {
	identity := credentialIdentity(submitted)
	if identity == "" {
		return -1
	}
	match := -1
	for i, item := range stored {
		if credentialIdentity(item) != identity {
			continue
		}
		if match >= 0 {
			return -1
		}
		match = i
	}
	return match
}

// credentialIdentity returns a JSON identity for the credentials in v, with
// every secret key removed. Fields next to a credential are ignored, so editing
// them does not break the match. It returns "" when v holds no credential.
func credentialIdentity(v any) string {
	identities := findCredentialIdentities(v)
	if len(identities) == 0 {
		return ""
	}
	b, err := json.Marshal(identities)
	if err != nil {
		return ""
	}
	return string(b)
}

// findCredentialIdentities collects the redacted JSON of every credential in v,
// sorted so an identity does not depend on map or slice order.
func findCredentialIdentities(v any) []string {
	var identities []string
	var walk func(any)
	walk = func(v any) {
		switch v := v.(type) {
		case map[string]any:
			if isCredential(v) {
				redacted, err := asJSONValue(v)
				if err != nil {
					return
				}
				dropSecretKeys(redacted)
				b, err := json.Marshal(redacted)
				if err != nil {
					return
				}
				identities = append(identities, string(b))
				return
			}
			for _, val := range v {
				walk(val)
			}
		case []any:
			for _, item := range v {
				walk(item)
			}
		}
	}
	walk(v)
	sort.Strings(identities)
	return identities
}

// hasSecretKey reports whether v carries a credential secret key at any depth.
func hasSecretKey(v any) bool {
	switch v := v.(type) {
	case map[string]any:
		for k, val := range v {
			if credentialSecretKeys[k] || hasSecretKey(val) {
				return true
			}
		}
	case []any:
		for _, item := range v {
			if hasSecretKey(item) {
				return true
			}
		}
	}
	return false
}
