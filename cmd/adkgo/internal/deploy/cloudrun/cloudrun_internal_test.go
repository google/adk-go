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
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func baseTriggerFlags() triggerConfigFlags {
	return triggerConfigFlags{
		maxRetries: 3,
		baseDelay:  1 * time.Second,
		maxDelay:   10 * time.Second,
		maxRuns:    100,
	}
}

func TestWriteTriggerArgs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		cfg     triggerConfigFlags
		want    []string
		notWant []string
	}{
		{
			name:    "no OIDC flags leaves the command as it was",
			cfg:     baseTriggerFlags(),
			want:    []string{`"--trigger_max_retries", "3"`, `"--trigger_max_concurrent_runs", "100"`},
			notWant: []string{"--trigger_oidc_audience", "--trigger_oidc_service_accounts"},
		},
		{
			name: "audience is forwarded",
			cfg: func() triggerConfigFlags {
				c := baseTriggerFlags()
				c.oidcAudience = "https://agent-abc.a.run.app/api/apps/x/trigger/pubsub"
				return c
			}(),
			want:    []string{`"--trigger_oidc_audience", "https://agent-abc.a.run.app/api/apps/x/trigger/pubsub"`},
			notWant: []string{"--trigger_oidc_service_accounts"},
		},
		{
			name: "allow-list is forwarded",
			cfg: func() triggerConfigFlags {
				c := baseTriggerFlags()
				c.oidcAudience = "https://agent-abc.a.run.app"
				c.oidcServiceAccounts = "push@p.iam.gserviceaccount.com,other@p.iam.gserviceaccount.com"
				return c
			}(),
			want: []string{
				`"--trigger_oidc_audience", "https://agent-abc.a.run.app"`,
				`"--trigger_oidc_service_accounts", "push@p.iam.gserviceaccount.com,other@p.iam.gserviceaccount.com"`,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var b strings.Builder
			writeTriggerArgs(&b, tt.cfg)
			got := b.String()

			for _, want := range tt.want {
				if !strings.Contains(got, want) {
					t.Errorf("writeTriggerArgs() = %q, want it to contain %q", got, want)
				}
			}
			for _, notWant := range tt.notWant {
				if strings.Contains(got, notWant) {
					t.Errorf("writeTriggerArgs() = %q, want it not to contain %q", got, notWant)
				}
			}
		})
	}
}

// The CMD line is a JSON array, so a flag value carrying a quote must not be
// able to close the string it sits in and add arguments of its own.
func TestWriteTriggerArgsEscapesFlagValues(t *testing.T) {
	t.Parallel()

	cfg := baseTriggerFlags()
	cfg.oidcAudience = `https://x", "--include_debug_api", "`

	var b strings.Builder
	b.WriteString(`["/app/agent", "web"`)
	writeTriggerArgs(&b, cfg)
	b.WriteString(`]`)

	var args []string
	if err := json.Unmarshal([]byte(b.String()), &args); err != nil {
		t.Fatalf("generated CMD is not valid JSON: %v\n%s", err, b.String())
	}
	for _, arg := range args {
		if arg == "--include_debug_api" {
			t.Errorf("flag value injected an extra argument into the container command: %v", args)
		}
	}
	if got := args[len(args)-1]; got != cfg.oidcAudience {
		t.Errorf("audience argument = %q, want %q", got, cfg.oidcAudience)
	}
}
