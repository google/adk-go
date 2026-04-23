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

package pubsub

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gorilla/mux"

	"google.golang.org/adk/cmd/launcher"
)

func TestParse(t *testing.T) {
	tests := []struct {
		name       string
		args       []string
		wantPrefix string
		wantRetry  int
		wantErr    bool
	}{
		{
			name:       "default values",
			args:       []string{},
			wantPrefix: "/api",
			wantRetry:  3,
			wantErr:    false,
		},
		{
			name:       "custom prefix and retries",
			args:       []string{"-path_prefix=/custom", "-trigger_max_retries=5"},
			wantPrefix: "/custom",
			wantRetry:  5,
			wantErr:    false,
		},
		{
			name:       "invalid retry count",
			args:       []string{"-trigger_max_retries=-1"},
			wantPrefix: "/api",
			wantRetry:  3,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := NewLauncher().(*pubsubLauncher)
			_, err := l.Parse(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("Parse() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}
			if l.config.pathPrefix != tt.wantPrefix {
				t.Errorf("Parse() pathPrefix = %v, want %v", l.config.pathPrefix, tt.wantPrefix)
			}
			if l.config.triggerMaxRetries != tt.wantRetry {
				t.Errorf("Parse() triggerMaxRetries = %v, want %v", l.config.triggerMaxRetries, tt.wantRetry)
			}
		})
	}
}

func TestSetupSubrouters(t *testing.T) {
	l := NewLauncher().(*pubsubLauncher)
	_, _ = l.Parse([]string{"-path_prefix=/api"})

	router := mux.NewRouter()
	config := &launcher.Config{}

	err := l.SetupSubrouters(router, config)
	if err != nil {
		t.Fatalf("SetupSubrouters() failed: %v", err)
	}

	// Verify route is registered
	req := httptest.NewRequest(http.MethodPost, "/api/apps/my-app/trigger/pubsub", nil)
	var match mux.RouteMatch
	if !router.Match(req, &match) {
		t.Errorf("SetupSubrouters() did not register expected route")
	}
}
