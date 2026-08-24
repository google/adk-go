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
		name           string
		flags          deployCloudRunFlags
		wantContains   []string
		wantNotContain []string
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
			wantContains: []string{
				"run", "deploy", "my-service",
				"--region", "us-central1",
				"--project", "my-project",
				"--set-secrets=GOOGLE_API_KEY=GOOGLE_API_KEY:latest",
			},
			wantNotContain: []string{
				"--service-account",
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
			wantContains: []string{
				"run", "deploy", "agent-service",
				"--region", "europe-west1",
				"--project", "eu-project",
				"--set-secrets=GOOGLE_API_KEY=CUSTOM_GEMINI_KEY:latest",
				"--service-account", "agent-sa@eu-project.iam.gserviceaccount.com",
			},
			wantNotContain: []string{},
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
			wantContains: []string{
				"run", "deploy", "no-secret-service",
				"--service-account", "runner-sa@test-proj.iam.gserviceaccount.com",
			},
			wantNotContain: []string{
				"--set-secrets",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.flags.buildDeployParams()

			for _, want := range tc.wantContains {
				if !slices.Contains(got, want) {
					t.Errorf("buildDeployParams() missing expected argument %q; got %v", want, got)
				}
			}

			for _, notWant := range tc.wantNotContain {
				for _, arg := range got {
					if arg == notWant || (len(notWant) > 0 && len(arg) >= len(notWant) && arg[:len(notWant)] == notWant) {
						t.Errorf("buildDeployParams() contains unexpected argument %q; got %v", arg, got)
					}
				}
			}
		})
	}
}
