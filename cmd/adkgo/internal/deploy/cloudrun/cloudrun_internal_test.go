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
	"os"
	"path/filepath"
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
// able to close the string it sits in and add arguments of its own. Both
// operator-supplied values go through the same jsonString call, so both are
// exercised here and the whole array is round-tripped through json.Unmarshal.
func TestWriteTriggerArgsEscapesFlagValues(t *testing.T) {
	t.Parallel()

	cfg := baseTriggerFlags()
	cfg.oidcAudience = `https://x", "--include_debug_api", "`
	cfg.oidcServiceAccounts = `evil@p.iam", "--another_injected_flag", "`

	var b strings.Builder
	b.WriteString(`["/app/agent", "web"`)
	writeTriggerArgs(&b, cfg)
	b.WriteString(`]`)

	var args []string
	if err := json.Unmarshal([]byte(b.String()), &args); err != nil {
		t.Fatalf("generated CMD is not valid JSON: %v\n%s", err, b.String())
	}
	for _, arg := range args {
		if arg == "--include_debug_api" || arg == "--another_injected_flag" {
			t.Errorf("flag value injected an extra argument into the container command: %v", args)
		}
	}
	// Each hostile value must survive verbatim as a single argument, immediately
	// after its own flag name, rather than being split.
	assertArgFollows(t, args, "--trigger_oidc_audience", cfg.oidcAudience)
	assertArgFollows(t, args, "--trigger_oidc_service_accounts", cfg.oidcServiceAccounts)
}

// assertArgFollows checks that value appears in args immediately after flag.
func assertArgFollows(t *testing.T, args []string, flag, value string) {
	t.Helper()
	for i, a := range args {
		if a == flag {
			if i+1 >= len(args) || args[i+1] != value {
				t.Errorf("%s argument = %q, want %q", flag, args[i+1:], value)
			}
			return
		}
	}
	t.Errorf("%s not present in %v", flag, args)
}

// TestValidateTriggerOIDCFlags pins the parse-time guards: a service-account
// allow-list without an audience is rejected rather than silently disabling
// verification, and a value that cannot be embedded safely in the generated
// Dockerfile is rejected before any filesystem work.
func TestValidateTriggerOIDCFlags(t *testing.T) {
	t.Parallel()

	withPubsub := func(t triggerConfigFlags) *deployCloudRunFlags {
		f := &deployCloudRunFlags{}
		f.cloudRun.pubsubTrigger = t
		return f
	}
	withEventarc := func(t triggerConfigFlags) *deployCloudRunFlags {
		f := &deployCloudRunFlags{}
		f.cloudRun.eventarcTrigger = t
		return f
	}
	accountsNoAudience := func() triggerConfigFlags {
		c := baseTriggerFlags()
		c.oidcServiceAccounts = "push@p.iam.gserviceaccount.com"
		return c
	}
	audienceBadUTF8 := func() triggerConfigFlags {
		c := baseTriggerFlags()
		c.oidcAudience = "https://x\xff"
		return c
	}
	audienceAndAccounts := func() triggerConfigFlags {
		c := baseTriggerFlags()
		c.oidcAudience = "https://agent-abc.a.run.app"
		c.oidcServiceAccounts = "push@p.iam.gserviceaccount.com"
		return c
	}

	tests := []struct {
		name    string
		flags   *deployCloudRunFlags
		wantErr string // substring; "" means no error
	}{
		{"pubsub accounts without audience", withPubsub(accountsNoAudience()), "--pubsub_oidc_service_accounts requires --pubsub_oidc_audience"},
		{"eventarc accounts without audience", withEventarc(accountsNoAudience()), "--eventarc_oidc_service_accounts requires --eventarc_oidc_audience"},
		{"pubsub audience with bad utf-8", withPubsub(audienceBadUTF8()), "--pubsub_oidc_audience"},
		{"eventarc audience with bad utf-8", withEventarc(audienceBadUTF8()), "--eventarc_oidc_audience"},
		{"pubsub audience plus allow-list is accepted", withPubsub(audienceAndAccounts()), ""},
		{"eventarc audience plus allow-list is accepted", withEventarc(audienceAndAccounts()), ""},
		{"no OIDC flags is accepted", withPubsub(baseTriggerFlags()), ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := tt.flags.validateTriggerOIDCFlags()
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("validateTriggerOIDCFlags() = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("validateTriggerOIDCFlags() = %v, want error containing %q", err, tt.wantErr)
			}
		})
	}
}

// TestPrepareDockerfileWiresEachTriggerAudience drives prepareDockerfile end to
// end and asserts that each trigger's own audience lands in its own section of
// the emitted CMD. writeTriggerArgs is called once per trigger off the
// receiver, so swapping which trigger a call site passes would surface here
// even though a direct writeTriggerArgs test would not.
func TestPrepareDockerfileWiresEachTriggerAudience(t *testing.T) {
	t.Parallel()

	const (
		pubsubAud   = "https://agent.example/api/apps/x/trigger/pubsub"
		eventarcAud = "https://agent.example/api/apps/x/trigger/eventarc"
	)

	f := &deployCloudRunFlags{}
	f.build.execFile = "agent"
	f.build.dockerfileBuildPath = filepath.Join(t.TempDir(), "Dockerfile")
	f.cloudRun.serverPort = 8080
	f.proxy.port = 8081
	f.cloudRun.pubsub = true
	f.cloudRun.eventarc = true
	f.cloudRun.pubsubTrigger = baseTriggerFlags()
	f.cloudRun.pubsubTrigger.oidcAudience = pubsubAud
	f.cloudRun.eventarcTrigger = baseTriggerFlags()
	f.cloudRun.eventarcTrigger.oidcAudience = eventarcAud

	if err := f.prepareDockerfile(); err != nil {
		t.Fatalf("prepareDockerfile() error: %v", err)
	}

	content, err := os.ReadFile(f.build.dockerfileBuildPath)
	if err != nil {
		t.Fatalf("reading generated Dockerfile: %v", err)
	}
	args := parseCMDArgs(t, string(content))

	pubsubIdx := indexOf(args, "pubsub")
	eventarcIdx := indexOf(args, "eventarc")
	if pubsubIdx < 0 || eventarcIdx < 0 || pubsubIdx > eventarcIdx {
		t.Fatalf("expected a pubsub section before an eventarc section, got %v", args)
	}

	pubsubSection := args[pubsubIdx:eventarcIdx]
	eventarcSection := args[eventarcIdx:]

	if !contains(pubsubSection, pubsubAud) {
		t.Errorf("pubsub section is missing its own audience %q: %v", pubsubAud, pubsubSection)
	}
	if contains(pubsubSection, eventarcAud) {
		t.Errorf("pubsub section carries the eventarc audience %q: %v", eventarcAud, pubsubSection)
	}
	if !contains(eventarcSection, eventarcAud) {
		t.Errorf("eventarc section is missing its own audience %q: %v", eventarcAud, eventarcSection)
	}
	if contains(eventarcSection, pubsubAud) {
		t.Errorf("eventarc section carries the pubsub audience %q: %v", pubsubAud, eventarcSection)
	}
}

// parseCMDArgs extracts the JSON array from the generated Dockerfile's CMD line.
func parseCMDArgs(t *testing.T, dockerfile string) []string {
	t.Helper()
	const marker = "CMD "
	i := strings.Index(dockerfile, marker)
	if i < 0 {
		t.Fatalf("no CMD line in generated Dockerfile:\n%s", dockerfile)
	}
	line := strings.TrimSpace(dockerfile[i+len(marker):])
	var args []string
	if err := json.Unmarshal([]byte(line), &args); err != nil {
		t.Fatalf("CMD is not a valid JSON array: %v\n%s", err, line)
	}
	return args
}

func indexOf(xs []string, target string) int {
	for i, x := range xs {
		if x == target {
			return i
		}
	}
	return -1
}

func contains(xs []string, target string) bool {
	return indexOf(xs, target) >= 0
}
