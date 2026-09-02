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

package cloudrun

import (
	"slices"
	"testing"
)

func TestBuildDeployParams(t *testing.T) {
	tests := []struct {
		name       string
		flags      deployCloudRunFlags
		wantParams []string
	}{
		{
			name: "default_secret_no_service_account",
			flags: deployCloudRunFlags{
				gcloud: gCloudFlags{
					region:      "us-central1",
					projectName: "my-project",
				},
				cloudRun: cloudRunServiceFlags{
					serviceName: "my-service",
					secretName:  "GOOGLE_API_KEY",
				},
			},
			wantParams: []string{
				"run", "deploy", "my-service",
				"--source", ".",
				"--region", "us-central1",
				"--project", "my-project",
				"--ingress", "all",
				"--no-allow-unauthenticated",
				"--set-secrets=GOOGLE_API_KEY=GOOGLE_API_KEY:latest",
			},
		},
		{
			name: "custom_service_account_and_custom_secret",
			flags: deployCloudRunFlags{
				gcloud: gCloudFlags{
					region:      "europe-west1",
					projectName: "eu-project",
				},
				cloudRun: cloudRunServiceFlags{
					serviceName:    "agent-service",
					serviceAccount: "agent-sa@eu-project.iam.gserviceaccount.com",
					secretName:     "CUSTOM_GEMINI_KEY",
				},
			},
			wantParams: []string{
				"run", "deploy", "agent-service",
				"--source", ".",
				"--region", "europe-west1",
				"--project", "eu-project",
				"--ingress", "all",
				"--no-allow-unauthenticated",
				"--set-secrets=GOOGLE_API_KEY=CUSTOM_GEMINI_KEY:latest",
				"--service-account", "agent-sa@eu-project.iam.gserviceaccount.com",
			},
		},
		{
			name: "empty_secret_name_skips_secret_mounting",
			flags: deployCloudRunFlags{
				gcloud: gCloudFlags{
					region:      "us-east1",
					projectName: "test-proj",
				},
				cloudRun: cloudRunServiceFlags{
					serviceName:    "no-secret-service",
					serviceAccount: "runner-sa@test-proj.iam.gserviceaccount.com",
					secretName:     "",
				},
			},
			wantParams: []string{
				"run", "deploy", "no-secret-service",
				"--source", ".",
				"--region", "us-east1",
				"--project", "test-proj",
				"--ingress", "all",
				"--no-allow-unauthenticated",
				"--service-account", "runner-sa@test-proj.iam.gserviceaccount.com",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.flags.buildDeployParams()

			if !slices.Equal(got, tc.wantParams) {
				t.Errorf("buildDeployParams() mismatch\ngot:  %v\nwant: %v", got, tc.wantParams)
			}
		})
	}
}
