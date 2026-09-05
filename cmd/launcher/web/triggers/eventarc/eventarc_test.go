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
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/cmd/launcher"
)

// TestSetupSubroutersWiresOIDCConfig checks that -trigger_oidc_audience actually
// reaches the controller, not just the launcher's own config struct. The
// startup warning reads the flag field, so config that never gets threaded
// through to TriggerConfig would suppress the warning and look correct while
// leaving the endpoint open. An unauthenticated request is rejected before any
// service on launcher.Config is touched, so the empty Config here is fine.
func TestSetupSubroutersWiresOIDCConfig(t *testing.T) {
	l := NewLauncher().(*eventarcLauncher)
	if _, err := l.Parse([]string{"-path_prefix=/api", "-trigger_oidc_audience=https://example-agent.example.com"}); err != nil {
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
		t.Errorf("unauthenticated request with -trigger_oidc_audience set: status = %d, want %d. "+
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
			args:    []string{"-trigger_oidc_service_accounts=sa@example.iam.gserviceaccount.com"},
			wantErr: true,
		},
		{
			name: "audience alone is allowed",
			args: []string{"-trigger_oidc_audience=https://example-agent.example.com"},
		},
		{
			name:         "service accounts are split and trimmed",
			args:         []string{"-trigger_oidc_audience=https://example-agent.example.com", "-trigger_oidc_service_accounts= a@example.iam.gserviceaccount.com , ,b@example.iam.gserviceaccount.com "},
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

// TestTriggerConfigWiresOIDC pins both halves of the OIDC configuration to the
// TriggerConfig that SetupSubrouters hands the controller, which is the only
// place the launcher can get either of them wrong.
//
// The end-to-end test above covers the audience half by observing a 401.
// The allow-list half cannot be reached the same way: proving a 403 needs a
// token that verifies, and the real idtoken.Validate would have to call
// Google's certificate endpoint, while the test seam that replaces it is
// unexported inside the triggers package by design. So this asserts on the
// config instead, and the warning in UserMessage reads the same function, so
// a setting that stops reaching the controller also stops being announced as
// enforced.
func TestTriggerConfigWiresOIDC(t *testing.T) {
	tests := []struct {
		name         string
		args         []string
		wantAudience string
		wantAccounts []string
	}{
		{
			name: "no OIDC flags leaves verification off",
			args: []string{},
		},
		{
			name:         "audience alone",
			args:         []string{"-trigger_oidc_audience=https://example-agent.example.com"},
			wantAudience: "https://example-agent.example.com",
		},
		{
			name: "audience and allow-list",
			args: []string{
				"-trigger_oidc_audience=https://example-agent.example.com",
				"-trigger_oidc_service_accounts=push@example-project.iam.gserviceaccount.com,other@example-project.iam.gserviceaccount.com",
			},
			wantAudience: "https://example-agent.example.com",
			wantAccounts: []string{
				"push@example-project.iam.gserviceaccount.com",
				"other@example-project.iam.gserviceaccount.com",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := NewLauncher().(*eventarcLauncher)
			if _, err := l.Parse(tt.args); err != nil {
				t.Fatalf("Parse() failed: %v", err)
			}

			oidc := l.triggerConfig().OIDC
			if tt.wantAudience == "" {
				if oidc != nil {
					t.Fatalf("triggerConfig().OIDC = %+v, want nil", oidc)
				}
				return
			}
			if oidc == nil {
				t.Fatal("triggerConfig().OIDC = nil; the audience is not reaching the controller")
			}
			if oidc.ExpectedAudience != tt.wantAudience {
				t.Errorf("ExpectedAudience = %q, want %q", oidc.ExpectedAudience, tt.wantAudience)
			}
			if !slices.Equal(oidc.AllowedServiceAccounts, tt.wantAccounts) {
				t.Errorf("AllowedServiceAccounts = %v, want %v; the allow-list is not reaching the controller", oidc.AllowedServiceAccounts, tt.wantAccounts)
			}
		})
	}
}

// The startup warning must follow the config that was built, not the raw flag
// strings: an operator who is told the allow-list is on must not be looking at
// an endpoint that accepts any holder of a Google-signed token.
func TestUserMessageWarnsFromBuiltConfig(t *testing.T) {
	tests := []struct {
		name     string
		args     []string
		wantWarn string
	}{
		{
			name:     "no audience",
			args:     []string{},
			wantWarn: "-trigger_oidc_audience is not set",
		},
		{
			name:     "audience without an allow-list",
			args:     []string{"-trigger_oidc_audience=https://example-agent.example.com"},
			wantWarn: "-trigger_oidc_service_accounts is not set",
		},
		{
			name: "audience and allow-list",
			args: []string{
				"-trigger_oidc_audience=https://example-agent.example.com",
				"-trigger_oidc_service_accounts=push@example-project.iam.gserviceaccount.com",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := NewLauncher().(*eventarcLauncher)
			if _, err := l.Parse(tt.args); err != nil {
				t.Fatalf("Parse() failed: %v", err)
			}

			var lines []string
			l.UserMessage("http://localhost:8080", func(v ...any) {
				lines = append(lines, fmt.Sprint(v...))
			})
			printed := strings.Join(lines, "\n")

			warned := strings.Contains(printed, "WARNING")
			if tt.wantWarn == "" {
				if warned {
					t.Errorf("UserMessage() warned with a full configuration:\n%s", printed)
				}
				return
			}
			if !strings.Contains(printed, tt.wantWarn) {
				t.Errorf("UserMessage() = %q, want a warning containing %q", printed, tt.wantWarn)
			}
		})
	}
}
