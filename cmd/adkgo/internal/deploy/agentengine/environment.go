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
	"fmt"
	"regexp"
	"strings"

	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"github.com/spf13/cobra"
)

func registerEnvironmentFlags(cmd *cobra.Command, f *agentEngineServiceFlags) {
	// The Go CLI uses repeated --env flags rather than Python's agent_config
	// env_vars, as requested in google/adk-go#848.
	cmd.PersistentFlags().StringArrayVar(&f.env, "env", nil, "Environment variable NAME=VALUE (repeatable). Overrides a default with the same name. On update, replaces the environment together with --default_env.")
	cmd.PersistentFlags().BoolVar(&f.defaultEnv, "default_env", true, "Include the default Agent Engine environment variables. Explicitly setting this flag replaces the environment on update; omit both environment flags to preserve it.")
}

var environmentNamePattern = regexp.MustCompile(`^[a-zA-Z_][a-zA-Z0-9_]*$`)

func (f *deployAgentEngineFlags) environmentVariables() ([]*aiplatformpb.EnvVar, error) {
	custom := make([]*aiplatformpb.EnvVar, 0, len(f.agentEngine.env))
	seen := make(map[string]bool)
	for i, raw := range f.agentEngine.env {
		name, value, ok := strings.Cut(raw, "=")
		if !ok || !environmentNamePattern.MatchString(name) {
			// Do not echo malformed input: it may contain a secret, including
			// when the user accidentally omitted NAME=.
			return nil, fmt.Errorf("--env entry %d must have the form NAME=VALUE with a name matching [a-zA-Z_][a-zA-Z0-9_]*", i+1)
		}
		if seen[name] {
			return nil, fmt.Errorf("--env entry %d repeats an environment variable name", i+1)
		}
		seen[name] = true
		custom = append(custom, &aiplatformpb.EnvVar{Name: name, Value: value})
	}

	var vars []*aiplatformpb.EnvVar
	if f.agentEngine.defaultEnv {
		defaults := []*aiplatformpb.EnvVar{
			{Name: "GOOGLE_CLOUD_REGION", Value: f.gcloud.region},
			{Name: "NUM_WORKERS", Value: "1"},
			{Name: "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", Value: "true"},
			{Name: "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", Value: "true"},
		}
		for _, env := range defaults {
			if !seen[env.Name] {
				vars = append(vars, env)
			}
		}
	}
	// EnvVar expands references to previously defined variables. Keep custom
	// flags in their original order, including overrides of a default.
	return append(vars, custom...), nil
}
