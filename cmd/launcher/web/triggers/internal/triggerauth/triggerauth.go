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
	"strings"

	"google.golang.org/adk/v2/server/authn"
)

// Flags are the -oidc_audience and -oidc_service_accounts flags.
type Flags struct {
	audience        string
	serviceAccounts string
}

// Register adds the flags to fs.
func (f *Flags) Register(fs *flag.FlagSet) {
	fs.StringVar(&f.audience, "oidc_audience", "", "Audience a Google-signed OIDC token must carry to call this "+
		"trigger endpoint: the token audience configured on the push subscription or Eventarc trigger. "+
		"Requires -oidc_service_accounts. When unset, the endpoint is unauthenticated.")
	fs.StringVar(&f.serviceAccounts, "oidc_service_accounts", "", "Comma-separated service account emails "+
		"allowed to call this trigger endpoint, matched against the token's verified email. "+
		"Requires -oidc_audience.")
}

// Authenticator builds the authenticator the flags describe. It returns nil
// when neither flag is set, which keeps the endpoint unauthenticated.
func (f *Flags) Authenticator() (authn.Authenticator, error) {
	if f.audience == "" && f.serviceAccounts == "" {
		return nil, nil
	}
	if f.serviceAccounts == "" {
		return nil, errors.New("-oidc_audience requires -oidc_service_accounts: the audience alone does not identify the caller")
	}
	if f.audience == "" {
		return nil, errors.New("-oidc_service_accounts requires -oidc_audience")
	}
	// Empty entries are kept so NewGoogleOIDC rejects a stray comma at startup.
	accounts := strings.Split(f.serviceAccounts, ",")
	for i := range accounts {
		accounts[i] = strings.TrimSpace(accounts[i])
	}
	return authn.NewGoogleOIDC(authn.GoogleOIDCConfig{
		Audience:               f.audience,
		AllowedServiceAccounts: accounts,
	})
}
