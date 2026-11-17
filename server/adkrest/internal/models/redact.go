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
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"

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

// MergeRedactedValue fills the credential secrets that a read strips from
// storedValue into submitted, and returns submitted. A client that edits a
// field next to a credential sends the credential back as the read returned it,
// without the secrets; writing that value back unchanged would erase them.
//
// Only omitted credential secret keys are filled in. submitted stays
// authoritative for every other field. A secret key sent as JSON null counts as
// omitted, because a read never returns a secret as null.
//
// A secret is restored only inside a credential that outbound redaction still
// treats as a credential, that still has the same authType and visible
// principal as the stored one, and that carries no secret of its own. Below
// that credential the walk restores at any depth, as a read strips at any
// depth. A credential the client re-points at another app, user, resource or
// scheme is a replacement, and so is one that sends any secret, so to clear one
// secret and keep the others, resend the others too. A credential under a
// credential field such as rawAuthCredential that has no authType is written as
// submitted: a read strips it, but the merge has no authType to match it by.
//
// When the same credential identity appears more than once in storedValue or
// more than once in submitted, the match would be a guess. Nothing is restored
// into those credentials, which leaves them as submitted. The check covers one
// stored value and its submitted value: the same identity under two different
// state keys is matched independently. Array entries that are not themselves
// credentials pair by index, so an insert, remove or reorder among them keeps
// the submitted value.
//
// It reports whether submitted was changed. submitted is changed in place; on
// error the returned value must not be used and submitted may already be
// partially restored. ctx must not be nil, and a cancellation is noticed at the
// latest after restoreContextCheckEvery visited nodes.
func MergeRedactedValue(ctx context.Context, storedValue, submitted any) (any, bool, error) {
	stored, err := asJSONValue(storedValue)
	if err != nil {
		return submitted, false, err
	}
	w := &restoreWalk{ctx: ctx}
	w.err = ctx.Err()
	ambiguous := ambiguousIdentities(w, stored, submitted)
	if w.err != nil {
		return submitted, false, w.err
	}
	merged, restored := restoreCredentialSecrets(w, stored, submitted, true, ambiguous)
	if w.err != nil {
		return submitted, false, w.err
	}
	return merged, restored, nil
}

// restoreWalk counts the nodes one merge visits and polls the context every
// restoreContextCheckEvery nodes, so a large value cannot hold the request
// longer than the client waits.
type restoreWalk struct {
	ctx     context.Context
	visited int
	err     error
}

// restoreContextCheckEvery is how many visited nodes pass between two context
// checks.
const restoreContextCheckEvery = 1024

// step counts one visited node and reports whether the walk should continue.
func (w *restoreWalk) step() bool {
	if w.err != nil {
		return false
	}
	w.visited++
	if w.visited%restoreContextCheckEvery == 0 {
		w.err = w.ctx.Err()
	}
	return w.err == nil
}

// ambiguousIdentities returns the credential identities that appear more than
// once in stored or more than once in submitted. Restoring secrets for such an
// identity would have to guess which stored credential the client meant, so
// nothing is restored into them.
func ambiguousIdentities(w *restoreWalk, stored, submitted any) map[string]bool {
	storedCounts := make(map[string]int)
	submittedCounts := make(map[string]int)
	countCredentialIdentities(w, stored, storedCounts)
	countCredentialIdentities(w, submitted, submittedCounts)
	ambiguous := make(map[string]bool)
	for id, n := range storedCounts {
		if n > 1 {
			ambiguous[id] = true
		}
	}
	for id, n := range submittedCounts {
		if n > 1 {
			ambiguous[id] = true
		}
	}
	return ambiguous
}

// countCredentialIdentities counts the credentials in v by identity. The walk
// counts every credential at any depth, including one nested inside another,
// because the restore walk matches those on their own too.
func countCredentialIdentities(w *restoreWalk, v any, counts map[string]int) {
	if !w.step() {
		return
	}
	switch v := v.(type) {
	case map[string]any:
		if isCredential(v) {
			counts[credentialIdentityOf(w, v)]++
		}
		for _, val := range v {
			countCredentialIdentities(w, val, counts)
		}
	case []any:
		for _, item := range v {
			countCredentialIdentities(w, item, counts)
		}
	}
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
// in the credential secret keys the submitted value omits. A key is a secret
// only inside a credential that may restore: allow says whether the value
// around it still belongs to the stored credential, and a credential narrows
// that to one that still has the same authType and visible principal, carries
// no secret of its own, and whose identity is not ambiguous. Other omitted
// fields are left out, so the submitted value remains a full replacement of the
// value outside credentials.
func restoreCredentialSecrets(w *restoreWalk, stored, submitted any, allow bool, ambiguous map[string]bool) (any, bool) {
	if !w.step() {
		return submitted, false
	}
	switch s := submitted.(type) {
	case map[string]any:
		st, ok := stored.(map[string]any)
		if !ok {
			return submitted, false
		}
		if isCredential(st) {
			// A client that changes authType or a visible identifying field is
			// replacing the credential; restoring secrets into it would attach
			// them to a different credential and, for a changed authType, make
			// the serializer leave them visible. Secrets inside a replaced
			// credential are left as submitted too.
			stIdent := credentialIdentityOf(w, st)
			allow = allow && isCredential(s) && !ambiguous[stIdent] &&
				stIdent == credentialIdentityOf(w, s) && !hasSecretKey(w, s)
		}
		restored := false
		if allow {
			for k, v := range st {
				if current, ok := s[k]; (!ok || current == nil) && credentialSecretKeys[k] {
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
			if merged, changed := restoreCredentialSecrets(w, sv, sub, allow, ambiguous); changed {
				s[k] = merged
				restored = true
			}
		}
		return s, restored
	case []any:
		st, ok := stored.([]any)
		if !ok {
			return submitted, false
		}
		// Match a submitted entry to the stored entry that holds the same
		// credential, so a reorder, append or remove keeps each credential's
		// own secrets. Only entries that are themselves credentials are matched
		// this way: computing an identity for a whole subtree here would repeat
		// the work at every level of nesting. Every other entry pairs by index,
		// so an insert, remove or reorder among them keeps the submitted value
		// instead of moving a secret to a neighbor.
		//
		// Matching by an ambiguous identity is harmless: the credential-level
		// check above refuses to restore secrets into it.
		storedByID := make(map[string]int, len(st))
		for j, item := range st {
			if !w.step() {
				return submitted, false
			}
			m, ok := item.(map[string]any)
			if !ok || !isCredential(m) {
				continue
			}
			storedByID[credentialIdentityOf(w, m)] = j
		}
		restored := false
		for i, item := range s {
			if !w.step() {
				return s, restored
			}
			if m, ok := item.(map[string]any); ok && isCredential(m) {
				j, ok := storedByID[credentialIdentityOf(w, m)]
				if !ok {
					continue
				}
				if merged, changed := restoreCredentialSecrets(w, st[j], item, allow, ambiguous); changed {
					s[i] = merged
					restored = true
				}
				continue
			}
			if i >= len(st) {
				continue
			}
			if merged, changed := restoreCredentialSecrets(w, st[i], item, allow, ambiguous); changed {
				s[i] = merged
				restored = true
			}
		}
		return s, restored
	default:
		return submitted, false
	}
}

// credentialIdentityOf returns the authType and the visible principal of a
// credential: the non-secret fields that name the account it is for. Two
// credentials with different identities are not the same credential, so a
// client that changes one of these fields is replacing the credential.
//
// The principal is every clientId and resourceRef, plus the fields each type
// shows in a read: an HTTP credential's scheme and username, and a service
// account's clientEmail and projectId. An API key shows no principal once
// redacted, so two API keys cannot be told apart; see ambiguousIdentities.
func credentialIdentityOf(w *restoreWalk, m map[string]any) string {
	credType := normalizedCredentialType(m)
	principal := make(map[string]bool)
	collectPrincipalFields(w, m, credType, principal)
	var b strings.Builder
	appendIdentityField(&b, credType)
	for _, field := range slices.Sorted(maps.Keys(principal)) {
		appendIdentityField(&b, field)
	}
	return b.String()
}

// appendIdentityField writes s with its length in front, so two different field
// lists never encode the same way.
func appendIdentityField(b *strings.Builder, s string) {
	b.WriteString(strconv.Itoa(len(s)))
	b.WriteByte(':')
	b.WriteString(s)
}

// principalFieldName returns the canonical name of the visible field that
// identifies a credential of type credType, or "" when key does not identify
// one. Both spellings of a field map to the same name, so a camelCase and a
// snake_case credential compare equal.
func principalFieldName(credType, key string) string {
	switch key {
	case "clientId", "client_id":
		return "clientId"
	case "resourceRef", "resource_ref":
		return "resourceRef"
	}
	switch credType {
	case "http":
		switch key {
		case "scheme", "username":
			return key
		}
	case "serviceaccount":
		switch key {
		case "clientEmail", "client_email":
			return "clientEmail"
		case "projectId", "project_id":
			return "projectId"
		}
	}
	return ""
}

// collectPrincipalFields adds every visible identifying field in v to fields.
// It walks the whole credential, so either spelling of a field anywhere in it
// identifies the same credential. A field's value is kept as JSON, so a value
// that is not a string still takes part in the identity.
func collectPrincipalFields(w *restoreWalk, v any, credType string, fields map[string]bool) {
	if !w.step() {
		return
	}
	switch v := v.(type) {
	case map[string]any:
		for k, val := range v {
			if name := principalFieldName(credType, k); name != "" {
				if encoded, err := json.Marshal(val); err == nil {
					fields[name+"\x01"+string(encoded)] = true
				}
			}
			collectPrincipalFields(w, val, credType, fields)
		}
	case []any:
		for _, item := range v {
			collectPrincipalFields(w, item, credType, fields)
		}
	}
}

// normalizedCredentialType returns authType in one spelling, so the camelCase,
// snake_case and upper-case forms compare equal.
func normalizedCredentialType(m map[string]any) string {
	for _, k := range []string{"authType", "auth_type"} {
		if name, ok := m[k].(string); ok {
			return strings.ToLower(strings.ReplaceAll(name, "_", ""))
		}
	}
	return ""
}

// hasSecretKey reports whether v carries a credential secret key with a value.
// A secret key set to JSON null counts as omitted, so it does not make the
// credential a replacement.
func hasSecretKey(w *restoreWalk, v any) bool {
	if !w.step() {
		return false
	}
	switch v := v.(type) {
	case map[string]any:
		for k, val := range v {
			if credentialSecretKeys[k] && val != nil {
				return true
			}
			if hasSecretKey(w, val) {
				return true
			}
		}
	case []any:
		return slices.ContainsFunc(v, func(item any) bool { return hasSecretKey(w, item) })
	}
	return false
}
