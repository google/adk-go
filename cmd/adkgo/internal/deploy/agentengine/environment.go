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
	"slices"
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
	values := make(map[string]string)
	if f.agentEngine.defaultEnv {
		values["GOOGLE_CLOUD_REGION"] = f.gcloud.region
		values["NUM_WORKERS"] = "1"
		values["GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY"] = "true"
		values["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
	}
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
		values[name] = value
	}

	names := make([]string, 0, len(values))
	for name := range values {
		names = append(names, name)
	}
	slices.Sort(names)
	vars := make([]*aiplatformpb.EnvVar, 0, len(names))
	for _, name := range names {
		vars = append(vars, &aiplatformpb.EnvVar{Name: name, Value: values[name]})
	}
	return vars, nil
}
