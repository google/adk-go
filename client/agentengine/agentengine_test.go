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

package agentengine

import (
	"strings"
	"testing"
)

func TestResourceName(t *testing.T) {
	tests := []struct {
		name         string
		cfg          Config
		wantName     string
		wantLocation string
		wantErr      string // substring; "" means no error
	}{
		{
			name:         "triple form",
			cfg:          Config{Project: "p", Location: "us-central1", ReasoningEngine: "123"},
			wantName:     "projects/p/locations/us-central1/reasoningEngines/123",
			wantLocation: "us-central1",
		},
		{
			name:         "full name form",
			cfg:          Config{Name: "projects/p/locations/europe-west1/reasoningEngines/456"},
			wantName:     "projects/p/locations/europe-west1/reasoningEngines/456",
			wantLocation: "europe-west1",
		},
		{
			name:         "full name wins over triple",
			cfg:          Config{Name: "projects/p/locations/europe-west1/reasoningEngines/456", Project: "x", Location: "y", ReasoningEngine: "z"},
			wantName:     "projects/p/locations/europe-west1/reasoningEngines/456",
			wantLocation: "europe-west1",
		},
		{name: "missing project", cfg: Config{Location: "l", ReasoningEngine: "r"}, wantErr: "required"},
		{name: "missing location", cfg: Config{Project: "p", ReasoningEngine: "r"}, wantErr: "required"},
		{name: "missing engine", cfg: Config{Project: "p", Location: "l"}, wantErr: "required"},
		{name: "malformed full name", cfg: Config{Name: "reasoningEngines/456"}, wantErr: "valid segments"},
		{name: "empty", cfg: Config{}, wantErr: "required"},
		// The location becomes the endpoint host, so anything that could
		// change the dial target is rejected.
		{name: "host and port in location", cfg: Config{Name: "projects/p/locations/127.0.0.1:443#/reasoningEngines/1"}, wantErr: "valid segments"},
		{name: "query marker in location", cfg: Config{Name: "projects/p/locations/us-central1?/reasoningEngines/1"}, wantErr: "valid segments"},
		{name: "uppercase location", cfg: Config{Project: "p", Location: "US-Central1", ReasoningEngine: "1"}, wantErr: "valid segments"},
		{name: "space in triple location", cfg: Config{Project: "p", Location: "us-central1 ", ReasoningEngine: "1"}, wantErr: "valid segments"},
		{name: "full name passed as engine", cfg: Config{Project: "p", Location: "us-central1", ReasoningEngine: "projects/p/locations/us-central1/reasoningEngines/1"}, wantErr: "valid segments"},
		{name: "slash in project", cfg: Config{Project: "p/q", Location: "us-central1", ReasoningEngine: "1"}, wantErr: "valid segments"},
		{
			name:         "domain-scoped project",
			cfg:          Config{Project: "google.com:my-proj", Location: "us-central1", ReasoningEngine: "1"},
			wantName:     "projects/google.com:my-proj/locations/us-central1/reasoningEngines/1",
			wantLocation: "us-central1",
		},
		{
			name:         "domain-scoped project in full name",
			cfg:          Config{Name: "projects/google.com:my-proj/locations/us-central1/reasoningEngines/1"},
			wantName:     "projects/google.com:my-proj/locations/us-central1/reasoningEngines/1",
			wantLocation: "us-central1",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotName, gotLocation, err := resourceName(tc.cfg)
			if tc.wantErr == "" && err != nil {
				t.Fatalf("resourceName(%+v) error = %v, want nil", tc.cfg, err)
			}
			if tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)) {
				t.Fatalf("resourceName(%+v) error = %v, want containing %q", tc.cfg, err, tc.wantErr)
			}
			if gotName != tc.wantName || gotLocation != tc.wantLocation {
				t.Errorf("resourceName(%+v) = (%q, %q), want (%q, %q)", tc.cfg, gotName, gotLocation, tc.wantName, tc.wantLocation)
			}
		})
	}
}

func TestNewRejectsInvalidConfig(t *testing.T) {
	if _, err := New(t.Context(), Config{}); err == nil {
		t.Fatal("New(empty Config) succeeded, want error")
	}
}
