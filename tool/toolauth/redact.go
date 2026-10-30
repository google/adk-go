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

package toolauth

import (
	"fmt"
	"io"
	"log/slog"
	"strings"
)

// The model types print their secrets as <redacted>, the fields adk-python
// marks repr=False in auth_credential.py, and every value in Extra, as its
// BaseModelWithConfig.__repr_args__ does. Format covers every verb fmt passes
// to it, nested values and pointers included. fmt prints %p of a value and %w
// of a non-error as bad verbs without calling Format, so those two show the
// secrets; go vet reports the %w case only. LogValue covers slog, whose JSON
// handler would otherwise call MarshalJSON and write the secrets. A value
// logged inside some other struct still goes through MarshalJSON.

// redactedText is printed in place of a secret that is set.
const redactedText = "<redacted>"

// redact replaces each non-empty string with redactedText.
func redact(fields ...*string) {
	for _, f := range fields {
		if *f != "" {
			*f = redactedText
		}
	}
}

// redactValues returns a copy of m with every value replaced by with, so that
// the keys still show.
func redactValues[V any](m map[string]V, with V) map[string]V {
	if len(m) == 0 {
		return m
	}
	out := make(map[string]V, len(m))
	for k := range m {
		out[k] = with
	}
	return out
}

// printPlain writes v in the format f asks for. v is a copy of the model type
// called name, with its secrets already redacted, under a local type plain
// with no Format method of its own; %#v names the model type instead of
// plain. Format has no way to report a write error, and fmt.State writes into
// fmt's own buffer, so there is none to handle.
func printPlain(f fmt.State, verb rune, name string, v any) {
	s := fmt.Sprintf(fmt.FormatString(f, verb), v)
	if rest, ok := strings.CutPrefix(s, "toolauth.plain"); ok {
		s = "toolauth." + name + rest
	}
	_, _ = io.WriteString(f, s)
}

// logValue is every model type's LogValue: the text its Format prints.
func logValue(v fmt.Formatter) slog.Value {
	return slog.StringValue(fmt.Sprintf("%+v", v))
}

// Format implements [fmt.Formatter] for every verb. It prints the credentials
// with Password, Token and the values in Extra shown as <redacted>.
func (c HTTPCredentials) Format(f fmt.State, verb rune) {
	type plain HTTPCredentials
	p := plain(c)
	redact(&p.Password, &p.Token)
	p.Extra = redactValues(p.Extra, any(redactedText))
	printPlain(f, verb, "HTTPCredentials", p)
}

// LogValue implements [slog.LogValuer] with the text Format prints.
func (c HTTPCredentials) LogValue() slog.Value { return logValue(c) }

// Format implements [fmt.Formatter] for every verb. It prints the scheme with
// the values in AdditionalHeaders and Extra shown as <redacted>.
func (h HTTPAuth) Format(f fmt.State, verb rune) {
	type plain HTTPAuth
	p := plain(h)
	p.AdditionalHeaders = redactValues(p.AdditionalHeaders, redactedText)
	p.Extra = redactValues(p.Extra, any(redactedText))
	printPlain(f, verb, "HTTPAuth", p)
}

// LogValue implements [slog.LogValuer] with the text Format prints.
func (h HTTPAuth) LogValue() slog.Value { return logValue(h) }

// Format implements [fmt.Formatter] for every verb. It prints the credential
// with ClientSecret, AuthResponseURI, AuthCode, AccessToken, RefreshToken,
// IDToken, CodeVerifier and the values in Extra shown as <redacted>.
func (o OAuth2Auth) Format(f fmt.State, verb rune) {
	type plain OAuth2Auth
	p := plain(o)
	redact(&p.ClientSecret, &p.AuthResponseURI, &p.AuthCode, &p.AccessToken, &p.RefreshToken, &p.IDToken, &p.CodeVerifier)
	p.Extra = redactValues(p.Extra, any(redactedText))
	printPlain(f, verb, "OAuth2Auth", p)
}

// LogValue implements [slog.LogValuer] with the text Format prints.
func (o OAuth2Auth) LogValue() slog.Value { return logValue(o) }

// Format implements [fmt.Formatter] for every verb. It prints the key with
// PrivateKeyID, PrivateKey and the values in Extra shown as <redacted>.
func (s ServiceAccountCredential) Format(f fmt.State, verb rune) {
	type plain ServiceAccountCredential
	p := plain(s)
	redact(&p.PrivateKeyID, &p.PrivateKey)
	p.Extra = redactValues(p.Extra, any(redactedText))
	printPlain(f, verb, "ServiceAccountCredential", p)
}

// LogValue implements [slog.LogValuer] with the text Format prints.
func (s ServiceAccountCredential) LogValue() slog.Value { return logValue(s) }

// Format implements [fmt.Formatter] for every verb. It prints the
// configuration with the values in Extra shown as <redacted>; the key in
// ServiceAccountCredential redacts itself.
func (s ServiceAccount) Format(f fmt.State, verb rune) {
	type plain ServiceAccount
	p := plain(s)
	p.Extra = redactValues(p.Extra, any(redactedText))
	printPlain(f, verb, "ServiceAccount", p)
}

// LogValue implements [slog.LogValuer] with the text Format prints.
func (s ServiceAccount) LogValue() slog.Value { return logValue(s) }

// Format implements [fmt.Formatter] for every verb. It prints the credential
// with APIKey and the values in Extra shown as <redacted>; the nested
// credentials redact themselves.
func (c Credential) Format(f fmt.State, verb rune) {
	type plain Credential
	p := plain(c)
	redact(&p.APIKey)
	p.Extra = redactValues(p.Extra, any(redactedText))
	printPlain(f, verb, "Credential", p)
}

// LogValue implements [slog.LogValuer] with the text Format prints.
func (c Credential) LogValue() slog.Value { return logValue(c) }

// Format implements [fmt.Formatter] for every verb. It prints the config with
// the values in Extra shown as <redacted>; the credentials redact themselves.
func (c Config) Format(f fmt.State, verb rune) {
	type plain Config
	p := plain(c)
	p.Extra = redactValues(p.Extra, any(redactedText))
	printPlain(f, verb, "Config", p)
}

// LogValue implements [slog.LogValuer] with the text Format prints.
func (c Config) LogValue() slog.Value { return logValue(c) }

// Format implements [fmt.Formatter] for every verb. It prints the arguments
// with the values in Extra shown as <redacted>; the config redacts itself.
func (a ToolArguments) Format(f fmt.State, verb rune) {
	type plain ToolArguments
	p := plain(a)
	p.Extra = redactValues(p.Extra, any(redactedText))
	printPlain(f, verb, "ToolArguments", p)
}

// LogValue implements [slog.LogValuer] with the text Format prints.
func (a ToolArguments) LogValue() slog.Value { return logValue(a) }
