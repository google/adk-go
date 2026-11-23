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
