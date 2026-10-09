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

// Package triggerauth holds the OIDC flags shared by the Pub/Sub and Eventarc
// trigger sublaunchers.
package triggerauth

import (
	"errors"
	"flag"
	"fmt"
	"strings"

	"google.golang.org/adk/v2/server/authn"
)

// Flags are the -oidc_audience and -oidc_service_accounts flags.
type Flags struct {
	audience        string
	serviceAccounts string
	fs              *flag.FlagSet
}

// Register adds the flags to fs.
func (f *Flags) Register(fs *flag.FlagSet) {
	f.fs = fs
	fs.StringVar(&f.audience, "oidc_audience", "", "Audience a Google-signed OIDC token must carry to call this "+
		"trigger endpoint: the token audience configured on the push subscription or Eventarc trigger. "+
		"Requires -oidc_service_accounts. When neither flag is given, the endpoint is unauthenticated, and an empty value "+
		"for either is refused. Like every flag of "+
		"this trigger, it must follow the trigger's own keyword: in \"pubsub eventarc -oidc_audience=...\" "+
		"it configures eventarc only.")
	fs.StringVar(&f.serviceAccounts, "oidc_service_accounts", "", "Comma-separated service account emails "+
		"allowed to call this trigger endpoint, matched against the token's verified email. "+
		"Requires -oidc_audience.")
}

// newGoogleOIDC is replaced in tests to see the config Authenticator builds,
// which the returned authenticator does not expose.
var newGoogleOIDC = authn.NewGoogleOIDC

// Authenticator builds the authenticator the flags describe. It returns nil
// when neither flag is passed, which keeps the endpoint unauthenticated.
func (f *Flags) Authenticator() (authn.Authenticator, error) {
	// A launch script reading an unset variable passes the flag empty, and
	// leaving the endpoint open for that would hide the mistake.
	for _, fl := range []struct{ name, value string }{{"oidc_audience", f.audience}, {"oidc_service_accounts", f.serviceAccounts}} {
		if fl.value == "" && f.passed(fl.name) {
			return nil, fmt.Errorf("-%s is empty: give it a value or leave it out", fl.name)
		}
	}
	if f.audience == "" && f.serviceAccounts == "" {
		return nil, nil
	}
	if f.serviceAccounts == "" {
		return nil, errors.New("-oidc_audience requires -oidc_service_accounts: the audience alone does not identify the caller")
	}
	if f.audience == "" {
		return nil, errors.New("-oidc_service_accounts requires -oidc_audience")
	}
	// Tokens are matched on the exact audience, so a padded one would start
	// cleanly and then reject every delivery.
	if f.audience != strings.TrimSpace(f.audience) {
		return nil, fmt.Errorf("-oidc_audience %q has surrounding whitespace", f.audience)
	}
	accounts := strings.Split(f.serviceAccounts, ",")
	for i := range accounts {
		accounts[i] = strings.TrimSpace(accounts[i])
		if accounts[i] == "" {
			return nil, fmt.Errorf("-oidc_service_accounts %q has an empty entry", f.serviceAccounts)
		}
	}
	return newGoogleOIDC(authn.GoogleOIDCConfig{
		Audience:               f.audience,
		AllowedServiceAccounts: accounts,
	})
}

func (f *Flags) passed(name string) bool {
	found := false
	if f.fs != nil {
		f.fs.Visit(func(fl *flag.Flag) { found = found || fl.Name == name })
	}
	return found
}
