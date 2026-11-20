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

// credentialTypeNames maps the authType spellings that mark a JSON object as a
// credential to one canonical name: the values, names and lower-case names of
// adk-python's AuthCredentialTypes.
var credentialTypeNames = map[string]string{
	"apiKey": "apikey", "API_KEY": "apikey", "api_key": "apikey",
	"http": "http", "HTTP": "http",
	"oauth2": "oauth2", "OAUTH2": "oauth2",
	"openIdConnect": "openidconnect", "OPEN_ID_CONNECT": "openidconnect", "open_id_connect": "openidconnect",
	"serviceAccount": "serviceaccount", "SERVICE_ACCOUNT": "serviceaccount", "service_account": "serviceaccount",
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
	return normalizedCredentialType(m) != ""
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
// Only omitted credential secret keys are filled in, and only inside a
// credential that outbound redaction still treats as a credential, that still
// has the same authType and visible principal as the stored one, and that
// carries no secret of its own. Above and below that credential nothing is
// filled in, so a value outside a credential stays exactly as submitted. A
// credential the client re-points at another app, user, resource or scheme is a
// replacement, and so is one that sends any secret. A credential under a
// credential field such as rawAuthCredential that has no authType cannot be
// matched, so nothing is restored into it either: it is written as submitted,
// as on main.
//
// A secret sent as JSON null counts as sent, not as omitted, so sending null
// clears the stored secret and signing out works as on main.
//
// When the same credential identity appears more than once in storedValue or
// more than once in submitted, the match would be a guess. Nothing is restored
// into those credentials, which leaves them as submitted. The check covers one
// stored value and its submitted value: the same identity under two different
// state keys is matched independently. Array entries that are not themselves
// credentials pair by index, so an insert, remove or reorder among them keeps
// the submitted value.
//
// Restoring the secrets on write is behavior adk-python's update_session
// (src/google/adk/cli/api_server.py) does not have: it writes the state delta as
// sent, which erases the secrets of a redacted credential a client sends back
// after editing a neighboring field. This merge keeps them because the read the
// client edited stripped the secrets and the client cannot resend them.
//
// It reports whether submitted was changed. submitted is changed in place; on
// error the returned value must not be used and submitted may already be
// partially restored. ctx must not be nil, and a cancellation is noticed at the
// latest after restoreContextCheckEvery visited nodes.
func MergeRedactedValue(ctx context.Context, storedValue, submitted any) (any, bool, error) {
	merged, restored, _, err := mergeRedactedValue(ctx, storedValue, submitted)
	return merged, restored, err
}

// mergeRedactedValue is MergeRedactedValue with the number of nodes the walk
// visited, which tests use to check that a value costs linear work.
func mergeRedactedValue(ctx context.Context, storedValue, submitted any) (any, bool, int, error) {
	stored, err := asJSONValue(storedValue)
	if err != nil {
		return submitted, false, 0, err
	}
	w := &restoreWalk{ctx: ctx}
	w.err = ctx.Err()
	if w.err != nil {
		return submitted, false, 0, w.err
	}
	storedInfo := &credentialAnalysis{counts: make(map[string]int)}
	submittedInfo := &credentialAnalysis{counts: make(map[string]int), hasSecret: make(map[string]bool)}
	analyzeCredentials(w, stored, storedInfo)
	if w.err == nil {
		analyzeCredentials(w, submitted, submittedInfo)
	}
	if w.err != nil {
		return submitted, false, w.visited, w.err
	}
	ambiguous := make(map[string]bool)
	for id, n := range storedInfo.counts {
		if n > 1 {
			ambiguous[id] = true
		}
	}
	for id, n := range submittedInfo.counts {
		if n > 1 {
			ambiguous[id] = true
		}
	}
	merged, restored := restoreCredentialSecrets(w, stored, submitted, false, ambiguous, submittedInfo.hasSecret)
	if w.err != nil {
		return submitted, false, w.visited, w.err
	}
	return merged, restored, w.visited, nil
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

// credentialAnalysis is what one merge learns about the credentials in a value:
// how often each identity appears, and, for the submitted value, which
// identities carry a secret. The merge uses it to find ambiguous identities and
// to decide whether a submitted credential is a replacement without walking a
// credential's subtree again for each of them.
type credentialAnalysis struct {
	counts    map[string]int
	hasSecret map[string]bool
}

// analyzeCredentials walks v once, counting every credential in it by identity.
// The walk counts every credential at any depth, including one nested inside
// another, because the restore walk matches those on their own too.
//
// When a.hasSecret is not nil, it also records per identity whether the
// credential carries a credential secret key at any depth. The flag is computed
// bottom-up in this same pass, so a value nested D levels deep is walked once,
// not once per level. A secret key counts whatever its value, including null.
//
// It reports whether v carries a secret key anywhere.
func analyzeCredentials(w *restoreWalk, v any, a *credentialAnalysis) bool {
	if !w.step() {
		return false
	}
	switch v := v.(type) {
	case map[string]any:
		id := ""
		if isCredential(v) {
			id = credentialIdentityOf(v)
			a.counts[id]++
		}
		hasSecret := false
		for k, val := range v {
			if credentialSecretKeys[k] {
				hasSecret = true
			}
			if analyzeCredentials(w, val, a) {
				hasSecret = true
			}
			if w.err != nil {
				return false
			}
		}
		if id != "" && a.hasSecret != nil {
			a.hasSecret[id] = hasSecret
		}
		return hasSecret
	case []any:
		hasSecret := false
		for _, item := range v {
			if analyzeCredentials(w, item, a) {
				hasSecret = true
			}
			if w.err != nil {
				return false
			}
		}
		return hasSecret
	}
	return false
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
// only inside a credential that may restore: allow says whether the walk is
// inside a stored credential that already passed the checks, and a credential
// turns it on only when the submitted value is still a credential with the same
// identity that carries no secret of its own and whose identity is not
// ambiguous. Everywhere else the submitted value stays authoritative, so the
// walk starts with allow false and nothing outside a credential is filled in.
// Other omitted fields are left out, so the submitted value remains a full
// replacement of the value outside credentials.
func restoreCredentialSecrets(w *restoreWalk, stored, submitted any, allow bool, ambiguous, hasSecret map[string]bool) (any, bool) {
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
			// the serializer leave them visible. A credential that carries a
			// secret of its own is a replacement too. Nothing is restored
			// inside a replaced or ambiguous credential, so skip its subtree.
			stIdent := credentialIdentityOf(st)
			if ambiguous[stIdent] || !isCredential(s) {
				return submitted, false
			}
			submittedIdent := credentialIdentityOf(s)
			if submittedIdent != stIdent || hasSecret[submittedIdent] {
				return submitted, false
			}
			allow = true
		}
		restored := false
		if allow {
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
			// A credential field's value that is not itself a credential has
			// no authType to match it by, so nothing is restored into it and it
			// is written as submitted, as on main. A real credential under the
			// field turns the restore back on when the walk enters it.
			childAllow := allow
			if credentialFields[k] {
				childAllow = false
			}
			if merged, changed := restoreCredentialSecrets(w, sv, sub, childAllow, ambiguous, hasSecret); changed {
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
		// so an insert, remove or reorder among them keeps the submitted value.
		//
		// An ambiguous identity restores nothing, so the walk skips it instead
		// of descending into its subtree for no result.
		storedByID := make(map[string]int, len(st))
		for j, item := range st {
			if !w.step() {
				return submitted, false
			}
			m, ok := item.(map[string]any)
			if !ok || !isCredential(m) {
				continue
			}
			id := credentialIdentityOf(m)
			if ambiguous[id] {
				continue
			}
			storedByID[id] = j
		}
		restored := false
		for i, item := range s {
			if !w.step() {
				return s, restored
			}
			if m, ok := item.(map[string]any); ok && isCredential(m) {
				id := credentialIdentityOf(m)
				if ambiguous[id] {
					continue
				}
				j, ok := storedByID[id]
				if !ok {
					continue
				}
				if merged, changed := restoreCredentialSecrets(w, st[j], item, allow, ambiguous, hasSecret); changed {
					s[i] = merged
					restored = true
				}
				continue
			}
			if i >= len(st) {
				continue
			}
			if merged, changed := restoreCredentialSecrets(w, st[i], item, allow, ambiguous, hasSecret); changed {
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
// Only the known fields of each type are read, at the positions the wire format
// puts them, and only scalar values take part. Walking the rest of the value
// would let one request allocate memory proportional to its nesting depth times
// its body size, would let a principal move to an unrelated path, and would
// count a secret header as part of the identity. An API key shows no principal
// once redacted, so two API keys cannot be told apart: an identity that appears
// more than once restores nothing.
func credentialIdentityOf(m map[string]any) string {
	credType := normalizedCredentialType(m)
	var b strings.Builder
	appendIdentityField(&b, credType)
	appendPrincipalField(&b, "resourceRef", scalarField(m, "resourceRef", "resource_ref"))
	switch credType {
	case "oauth2", "openidconnect":
		if oauth, ok := objectField(m, "oauth2"); ok {
			appendPrincipalField(&b, "clientId", scalarField(oauth, "clientId", "client_id"))
		}
	case "http":
		if httpAuth, ok := objectField(m, "http"); ok {
			appendPrincipalField(&b, "scheme", scalarField(httpAuth, "scheme"))
			if credentials, ok := objectField(httpAuth, "credentials"); ok {
				appendPrincipalField(&b, "username", scalarField(credentials, "username"))
			}
		}
	case "serviceaccount":
		if account, ok := objectField(m, "serviceAccount", "service_account"); ok {
			if key, ok := objectField(account, "serviceAccountCredential", "service_account_credential"); ok {
				appendPrincipalField(&b, "clientId", scalarField(key, "clientId", "client_id"))
				appendPrincipalField(&b, "clientEmail", scalarField(key, "clientEmail", "client_email"))
				appendPrincipalField(&b, "projectId", scalarField(key, "projectId", "project_id"))
			}
		}
	}
	return b.String()
}

// objectField returns the object held under the first spelling in keys that m
// holds, or false when the field is absent or is not an object.
func objectField(m map[string]any, keys ...string) (map[string]any, bool) {
	for _, key := range keys {
		if v, ok := m[key]; ok {
			obj, ok := v.(map[string]any)
			return obj, ok
		}
	}
	return nil, false
}

// appendPrincipalField writes one named principal value to an identity. It
// writes nothing when value is empty, which is how scalarField reports an
// absent or non-scalar field.
func appendPrincipalField(b *strings.Builder, name, value string) {
	if value == "" {
		return
	}
	appendIdentityField(b, name)
	appendIdentityField(b, value)
}

// scalarField returns the encoded scalar held under the first spelling in keys
// that m holds. It reports "" when the field is absent and when its value is
// not a scalar: only a scalar can name a credential, and a read shows only the
// identity it encoded.
func scalarField(m map[string]any, keys ...string) string {
	for _, key := range keys {
		v, ok := m[key]
		if !ok {
			continue
		}
		encoded, ok := encodeIdentityScalar(v)
		if !ok {
			return ""
		}
		return encoded
	}
	return ""
}

// encodeIdentityScalar encodes a JSON scalar so values of different JSON types
// never compare equal and numbers keep the spelling they arrived with.
func encodeIdentityScalar(v any) (string, bool) {
	switch v := v.(type) {
	case string:
		return "s:" + v, true
	case json.Number:
		return "n:" + v.String(), true
	case float64:
		return "n:" + strconv.FormatFloat(v, 'g', -1, 64), true
	case int:
		return "n:" + strconv.Itoa(v), true
	case int64:
		return "n:" + strconv.FormatInt(v, 10), true
	case bool:
		return "b:" + strconv.FormatBool(v), true
	}
	return "", false
}

// appendIdentityField writes s with its length in front, so two different field
// lists never encode the same way.
func appendIdentityField(b *strings.Builder, s string) {
	b.WriteString(strconv.Itoa(len(s)))
	b.WriteByte(':')
	b.WriteString(s)
}

// normalizedCredentialType returns the canonical authType of the credential in
// m, so the camelCase, snake_case and upper-case spellings compare equal. A
// spelling that is not a credential type is skipped, as isCredential does, so
// an unknown authType on one spelling cannot stand in for a known type on the
// other.
func normalizedCredentialType(m map[string]any) string {
	for _, k := range []string{"authType", "auth_type"} {
		if name, ok := m[k].(string); ok {
			if canonical, ok := credentialTypeNames[name]; ok {
				return canonical
			}
		}
	}
	return ""
}
