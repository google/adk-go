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

import "testing"

func TestParseSecretEnvVars(t *testing.T) {
	got, err := parseSecretEnvVars([]string{
		"GOOGLE_API_KEY=gemini-key:latest",
		"SLACK_TOKEN=slack-token:5",
	})
	if err != nil {
		t.Fatalf("parseSecretEnvVars() returned error: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("parseSecretEnvVars() returned %d secrets, want 2", len(got))
	}

	if got[0].Name != "GOOGLE_API_KEY" {
		t.Errorf("first env name = %q, want GOOGLE_API_KEY", got[0].Name)
	}
	if got[0].SecretRef.GetSecret() != "gemini-key" {
		t.Errorf("first secret = %q, want gemini-key", got[0].SecretRef.GetSecret())
	}
	if got[0].SecretRef.GetVersion() != "latest" {
		t.Errorf("first version = %q, want latest", got[0].SecretRef.GetVersion())
	}

	if got[1].Name != "SLACK_TOKEN" {
		t.Errorf("second env name = %q, want SLACK_TOKEN", got[1].Name)
	}
	if got[1].SecretRef.GetSecret() != "slack-token" {
		t.Errorf("second secret = %q, want slack-token", got[1].SecretRef.GetSecret())
	}
	if got[1].SecretRef.GetVersion() != "5" {
		t.Errorf("second version = %q, want 5", got[1].SecretRef.GetVersion())
	}
}

func TestParseSecretEnvVarsRejectsInvalidInput(t *testing.T) {
	tests := []string{
		"",
		"GOOGLE_API_KEY",
		"=gemini-key:latest",
		"GOOGLE_API_KEY=:latest",
		"GOOGLE_API_KEY=gemini-key:",
	}
	for _, tc := range tests {
		t.Run(tc, func(t *testing.T) {
			if _, err := parseSecretEnvVars([]string{tc}); err == nil {
				t.Fatalf("parseSecretEnvVars(%q) returned nil error, want error", tc)
			}
		})
	}
}

func TestDeploymentSpecUsesConfiguredSecrets(t *testing.T) {
	flags := &deployAgentEngineFlags{
		gcloud: gCloudFlags{
			region: "us-central1",
		},
		agentEngine: agentEngineServiceFlags{
			secrets: []string{"GOOGLE_API_KEY=gemini-key:latest"},
		},
	}

	spec, err := flags.deploymentSpec()
	if err != nil {
		t.Fatalf("deploymentSpec() returned error: %v", err)
	}
	if len(spec.Env) == 0 {
		t.Fatalf("deploymentSpec() returned no env vars")
	}
	if spec.Env[0].GetName() != "GOOGLE_CLOUD_REGION" || spec.Env[0].GetValue() != "us-central1" {
		t.Fatalf("first env = %s:%s, want GOOGLE_CLOUD_REGION:us-central1", spec.Env[0].GetName(), spec.Env[0].GetValue())
	}
	if len(spec.SecretEnv) != 1 {
		t.Fatalf("deploymentSpec() returned %d secret env vars, want 1", len(spec.SecretEnv))
	}
	if spec.SecretEnv[0].GetName() != "GOOGLE_API_KEY" {
		t.Errorf("secret env name = %q, want GOOGLE_API_KEY", spec.SecretEnv[0].GetName())
	}
	if spec.SecretEnv[0].GetSecretRef().GetSecret() != "gemini-key" {
		t.Errorf("secret ref = %q, want gemini-key", spec.SecretEnv[0].GetSecretRef().GetSecret())
	}
}

func TestDeploymentSpecAllowsNoSecrets(t *testing.T) {
	flags := &deployAgentEngineFlags{}

	spec, err := flags.deploymentSpec()
	if err != nil {
		t.Fatalf("deploymentSpec() returned error: %v", err)
	}
	if len(spec.SecretEnv) != 0 {
		t.Fatalf("deploymentSpec() returned %d secret env vars, want 0", len(spec.SecretEnv))
	}
}
