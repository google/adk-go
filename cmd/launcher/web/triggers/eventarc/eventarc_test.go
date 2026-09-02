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

package eventarc

import (
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/cmd/launcher"
)

// TestSetupSubroutersWiresOIDCConfig checks that -oidc_audience actually
// reaches the controller, not just the launcher's own config struct. The
// startup warning reads the flag field, so config that never gets threaded
// through to TriggerConfig would suppress the warning and look correct while
// leaving the endpoint open. An unauthenticated request is rejected before any
// service on launcher.Config is touched, so the empty Config here is fine.
func TestSetupSubroutersWiresOIDCConfig(t *testing.T) {
	l := NewLauncher().(*eventarcLauncher)
	if _, err := l.Parse([]string{"-path_prefix=/api", "-oidc_audience=https://example-agent.example.com"}); err != nil {
		t.Fatalf("Parse() failed: %v", err)
	}

	router := mux.NewRouter()
	if err := l.SetupSubrouters(router, &launcher.Config{}); err != nil {
		t.Fatalf("SetupSubrouters() failed: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/apps/my-app/trigger/eventarc", nil)
	rr := httptest.NewRecorder()
	router.ServeHTTP(rr, req)

	if rr.Code != http.StatusUnauthorized {
		t.Errorf("unauthenticated request with -oidc_audience set: status = %d, want %d. "+
			"The audience is likely not being passed through to the trigger controller. Body: %s", rr.Code, http.StatusUnauthorized, rr.Body.String())
	}
}

func TestParseOIDCFlags(t *testing.T) {
	tests := []struct {
		name         string
		args         []string
		wantErr      bool
		wantAccounts []string
	}{
		{
			name:    "service accounts without an audience is rejected",
			args:    []string{"-oidc_service_accounts=sa@example.iam.gserviceaccount.com"},
			wantErr: true,
		},
		{
			name: "audience alone is allowed",
			args: []string{"-oidc_audience=https://example-agent.example.com"},
		},
		{
			name:         "service accounts are split and trimmed",
			args:         []string{"-oidc_audience=https://example-agent.example.com", "-oidc_service_accounts= a@example.iam.gserviceaccount.com , ,b@example.iam.gserviceaccount.com "},
			wantAccounts: []string{"a@example.iam.gserviceaccount.com", "b@example.iam.gserviceaccount.com"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := NewLauncher().(*eventarcLauncher)
			_, err := l.Parse(tt.args)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Parse() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if got := splitList(l.config.oidcSvcAccounts); !slices.Equal(got, tt.wantAccounts) {
				t.Errorf("service accounts = %v, want %v", got, tt.wantAccounts)
			}
		})
	}
}
