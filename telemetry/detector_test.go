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

// Package telemetry implements the open telemetry in ADK.
package telemetry

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/resource"
	"golang.org/x/oauth2/google"
)

func TestProjectIDDetector(t *testing.T) {
	testCases := []struct {
		name          string
		cfg           *config
		envVar        string
		wantProjectID string
		wantErr       bool
	}{
		{
			name: "projectID from config",
			cfg: &config{
				resourceProjectID: "config-project",
				credentials:       &google.Credentials{ProjectID: "cred-project"},
			},
			envVar:        "env-project",
			wantProjectID: "config-project",
		},
		{
			name: "projectID from credentials",
			cfg: &config{
				credentials: &google.Credentials{ProjectID: "cred-project"},
			},
			envVar:        "env-project",
			wantProjectID: "cred-project",
		},
		{
			name:          "projectID from env var",
			cfg:           &config{},
			envVar:        "env-project",
			wantProjectID: "env-project",
		},
		{
			name: "no projectID",
			cfg: &config{
				credentials: &google.Credentials{},
			},
			wantProjectID: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.envVar != "" {
				t.Setenv("GOOGLE_CLOUD_PROJECT", tc.envVar)
			}

			detector := newProjectIDDetector(tc.cfg)
			res, err := detector.Detect(context.Background())
			if (err != nil) != tc.wantErr {
				t.Fatalf("Detect() error = %v, wantErr %v", err, tc.wantErr)
			}

			var expected *resource.Resource
			if tc.wantProjectID != "" {
				expected = resource.NewWithAttributes(
					"",
					attribute.String("gcp.project_id", tc.wantProjectID),
				)
			}

			if diff := cmp.Diff(expected, res); diff != "" {
				t.Errorf("Detect() returned diff (-want +got):\n%s", diff)
			}
		})
	}
}
