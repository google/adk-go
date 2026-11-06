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

package mcptoolset

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/v2/auth"
)

const testEndpoint = "https://mcp.example/mcp"

func TestNewRequiresConnection(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{name: "empty config"},
		{
			name: "client without transport",
			cfg: Config{
				Client: mcp.NewClient(&mcp.Implementation{Name: "test", Version: "1"}, nil),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts, err := New(tt.cfg)
			if err == nil {
				t.Fatal("New() succeeded without a transport or endpoint")
			}
			if ts != nil {
				t.Error("New() returned a toolset on configuration error")
			}
		})
	}
}

func TestBuildTransport(t *testing.T) {
	// Shared inputs for the wrap/passthrough cases. buildTransport must never
	// mutate a caller-supplied transport or client, so sharing these across
	// cases is safe (and is itself asserted below).
	base := &http.Client{Timeout: 7 * time.Second}
	callerStreamable := &mcp.StreamableClientTransport{
		Endpoint:             testEndpoint,
		HTTPClient:           base,
		MaxRetries:           9,
		DisableStandaloneSSE: true,
	}
	callerSSE := &mcp.SSEClientTransport{Endpoint: testEndpoint, HTTPClient: base}
	callerCommand := &mcp.CommandTransport{}
	// sentinelRT stands in for a caller's existing HTTP transport, to assert the
	// auth transport chains onto it rather than replacing it.
	sentinelRT := &stubRoundTripper{}

	tests := []struct {
		name    string
		cfg     Config
		wantErr bool
		check   func(t *testing.T, tr mcp.Transport)
	}{
		{
			name: "endpoint with auth builds a streamable client carrying the auth transport",
			cfg:  Config{Endpoint: testEndpoint, Auth: auth.StaticToken("tok")},
			check: func(t *testing.T, tr mcp.Transport) {
				st := mustStreamable(t, tr)
				if st.Endpoint != testEndpoint {
					t.Errorf("Endpoint = %q, want %q", st.Endpoint, testEndpoint)
				}
				wantsAuthTransport(t, st.HTTPClient)
			},
		},
		{
			name: "auth wraps a caller streamable transport on a copy, without mutating it",
			cfg:  Config{Transport: callerStreamable, Auth: auth.StaticToken("tok")},
			check: func(t *testing.T, tr mcp.Transport) {
				st := mustStreamable(t, tr)
				// Non-auth fields and base client settings are preserved on the copy.
				if st.Endpoint != testEndpoint || st.MaxRetries != 9 || !st.DisableStandaloneSSE {
					t.Errorf("preserved fields lost: %+v", st)
				}
				wantsAuthTransport(t, st.HTTPClient)
				if st.HTTPClient.Timeout != 7*time.Second {
					t.Errorf("client Timeout = %v, want 7s (base settings should be copied)", st.HTTPClient.Timeout)
				}
				// The caller's transport and client must be untouched.
				if st == callerStreamable {
					t.Error("caller transport was not copied")
				}
				if st.HTTPClient == base {
					t.Error("caller HTTPClient was not copied")
				}
				if callerStreamable.HTTPClient != base || base.Transport != nil {
					t.Error("caller's transport/client was mutated")
				}
			},
		},
		{
			name: "auth chains onto the caller's existing HTTP transport",
			cfg: Config{
				Transport: &mcp.StreamableClientTransport{
					Endpoint:   testEndpoint,
					HTTPClient: &http.Client{Transport: sentinelRT},
				},
				Auth: auth.StaticToken("tok"),
			},
			check: func(t *testing.T, tr mcp.Transport) {
				st := mustStreamable(t, tr)
				at, ok := st.HTTPClient.Transport.(*auth.Transport)
				if !ok {
					t.Fatalf("HTTPClient.Transport = %T, want *auth.Transport", st.HTTPClient.Transport)
				}
				trk, ok := at.Base.(*trackingTransport)
				if !ok {
					t.Fatalf("auth.Transport.Base = %T, want *trackingTransport", at.Base)
				}
				if trk.base != sentinelRT {
					t.Errorf("trackingTransport.base = %v, want the caller's original transport", trk.base)
				}
			},
		},
		{
			name:    "auth with a non-streamable transport is an error",
			cfg:     Config{Transport: &mcp.CommandTransport{}, Auth: auth.StaticToken("tok")},
			wantErr: true,
		},
		{
			name:    "auth without an endpoint or transport is an error",
			cfg:     Config{Auth: auth.StaticToken("tok")},
			wantErr: true,
		},
		{
			name: "endpoint without auth builds a streamable client carrying only the tracking transport",
			cfg:  Config{Endpoint: testEndpoint},
			check: func(t *testing.T, tr mcp.Transport) {
				st := mustStreamable(t, tr)
				if st.Endpoint != testEndpoint {
					t.Errorf("Endpoint = %q, want %q", st.Endpoint, testEndpoint)
				}
				if st.HTTPClient == nil {
					t.Fatal("HTTPClient = nil, want a client carrying the tracking transport")
				}
				if _, ok := st.HTTPClient.Transport.(*trackingTransport); !ok {
					t.Errorf("HTTPClient.Transport = %T, want *trackingTransport", st.HTTPClient.Transport)
				}
			},
		},
		{
			name: "SSE transport is copied without mutating the caller's",
			cfg:  Config{Transport: callerSSE},
			check: func(t *testing.T, tr mcp.Transport) {
				sse, ok := tr.(*mcp.SSEClientTransport)
				if !ok {
					t.Fatalf("transport = %T, want *mcp.SSEClientTransport", tr)
				}
				if sse == callerSSE || sse.HTTPClient == base {
					t.Error("caller SSE transport or its client was not copied")
				}
				if sse.Endpoint != testEndpoint || sse.HTTPClient.Timeout != 7*time.Second {
					t.Errorf("preserved fields lost: endpoint %q, timeout %v", sse.Endpoint, sse.HTTPClient.Timeout)
				}
				if callerSSE.HTTPClient != base || base.Transport != nil {
					t.Error("caller's SSE transport/client was mutated")
				}
			},
		},
		{
			name:    "auth with an SSE transport is an error",
			cfg:     Config{Transport: &mcp.SSEClientTransport{Endpoint: testEndpoint}, Auth: auth.StaticToken("tok")},
			wantErr: true,
		},
		{
			name: "streamable transport without auth is copied without mutating the caller's",
			cfg:  Config{Transport: callerStreamable},
			check: func(t *testing.T, tr mcp.Transport) {
				st := mustStreamable(t, tr)
				if st == callerStreamable || st.HTTPClient == base {
					t.Error("caller streamable transport or its client was not copied")
				}
				if st.Endpoint != testEndpoint || st.MaxRetries != 9 || !st.DisableStandaloneSSE || st.HTTPClient.Timeout != 7*time.Second {
					t.Errorf("preserved fields lost: %+v, timeout %v", st, st.HTTPClient.Timeout)
				}
				if callerStreamable.HTTPClient != base || base.Transport != nil {
					t.Error("caller's streamable transport/client was mutated")
				}
			},
		},
		{
			name:    "typed-nil streamable transport is an error",
			cfg:     Config{Transport: (*mcp.StreamableClientTransport)(nil)},
			wantErr: true,
		},
		{
			name:    "typed-nil streamable transport with auth is an error",
			cfg:     Config{Transport: (*mcp.StreamableClientTransport)(nil), Auth: auth.StaticToken("tok")},
			wantErr: true,
		},
		{
			name:    "typed-nil SSE transport is an error",
			cfg:     Config{Transport: (*mcp.SSEClientTransport)(nil)},
			wantErr: true,
		},
		{
			name: "a non-HTTP transport is passed through unchanged",
			cfg:  Config{Transport: callerCommand},
			check: func(t *testing.T, tr mcp.Transport) {
				if tr != callerCommand {
					t.Errorf("transport = %v, want the caller's transport unchanged", tr)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tr, err := buildTransport(tt.cfg)
			if tt.wantErr {
				if err == nil {
					t.Fatal("buildTransport() = nil error, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("buildTransport() error = %v", err)
			}
			tt.check(t, tr)
		})
	}
}

func mustStreamable(t *testing.T, tr mcp.Transport) *mcp.StreamableClientTransport {
	t.Helper()
	st, ok := tr.(*mcp.StreamableClientTransport)
	if !ok {
		t.Fatalf("transport = %T, want *mcp.StreamableClientTransport", tr)
	}
	return st
}

func wantsAuthTransport(t *testing.T, c *http.Client) {
	t.Helper()
	if c == nil {
		t.Fatal("HTTPClient = nil, want a client carrying the auth transport")
	}
	if _, ok := c.Transport.(*auth.Transport); !ok {
		t.Errorf("HTTPClient.Transport = %T, want *auth.Transport", c.Transport)
	}
}

// stubRoundTripper is a no-op http.RoundTripper used as a sentinel.
type stubRoundTripper struct{}

func (*stubRoundTripper) RoundTrip(*http.Request) (*http.Response, error) { return nil, nil }

// TestCommandTransportReconnect checks that a CommandTransport can connect
// more than once. An *exec.Cmd can be started only once, so the first connect
// uses the caller's command and later ones must start a copy. Each start of
// the helper writes its working directory to Stderr, which shows that every
// connect ran the caller's Path, Args, Env, Dir and Stderr.
func TestCommandTransportReconnect(t *testing.T) {
	dir := t.TempDir()
	stderr, err := os.Create(filepath.Join(t.TempDir(), "stderr"))
	if err != nil {
		t.Fatal(err)
	}
	defer stderr.Close()

	cmd := exec.Command(os.Args[0], "-test.run=^TestCommandTransportHelper$")
	cmd.Env = append(os.Environ(), "MCPTOOLSET_TEST_HELPER=1")
	cmd.Dir = dir
	cmd.Stderr = stderr
	tr, err := buildTransport(Config{Transport: &mcp.CommandTransport{Command: cmd}})
	if err != nil {
		t.Fatalf("buildTransport() error = %v", err)
	}
	for i := range 3 {
		conn, err := tr.Connect(t.Context())
		if err != nil {
			t.Fatalf("Connect #%d error = %v", i+1, err)
		}
		// The first connect must run the caller's own command, so settings
		// that cannot be copied (such as a CommandContext context) still apply.
		if i == 0 && cmd.Process == nil {
			t.Error("first Connect did not start the caller's command")
		}
		if err := conn.Close(); err != nil {
			t.Fatalf("Close #%d error = %v", i+1, err)
		}
	}

	out, err := os.ReadFile(stderr.Name())
	if err != nil {
		t.Fatal(err)
	}
	realDir, err := filepath.EvalSymlinks(dir)
	if err != nil {
		t.Fatal(err)
	}
	want := strings.Repeat("helper started in "+realDir+"\n", 3)
	if got := string(out); got != want {
		t.Errorf("helper output = %q, want %q", got, want)
	}
}

// TestCommandTransportReconnectKeepsSettings checks that the command a
// reconnect starts has every setting of the caller's command and transport.
func TestCommandTransportReconnectKeepsSettings(t *testing.T) {
	stdin := strings.NewReader("in")
	stderr := new(strings.Builder)
	extra := []*os.File{os.Stderr}
	attr := &syscall.SysProcAttr{}
	lookErr := errors.New("lookup failed")
	cmd := &exec.Cmd{
		Path:        "/usr/bin/server",
		Args:        []string{"server", "--flag"},
		Env:         []string{"A=1"},
		Dir:         "/srv",
		Stdin:       stdin,
		Stderr:      stderr,
		ExtraFiles:  extra,
		SysProcAttr: attr,
		WaitDelay:   3 * time.Second,
		Err:         lookErr,
	}
	ct := newCommandTransport(&mcp.CommandTransport{Command: cmd, TerminateDuration: 7 * time.Second})

	tr := ct.reconnectTransport()
	got := tr.Command
	if got == cmd {
		t.Fatal("reconnect reuses the caller's command, want a copy")
	}
	for _, c := range []struct {
		name string
		ok   bool
	}{
		{"Path", got.Path == cmd.Path},
		{"Args", slices.Equal(got.Args, cmd.Args)},
		{"Env", slices.Equal(got.Env, cmd.Env)},
		{"Dir", got.Dir == cmd.Dir},
		{"Stdin", got.Stdin == stdin},
		{"Stderr", got.Stderr == stderr},
		{"ExtraFiles", slices.Equal(got.ExtraFiles, extra)},
		{"SysProcAttr", got.SysProcAttr == attr},
		{"WaitDelay", got.WaitDelay == cmd.WaitDelay},
		{"Err", got.Err == lookErr},
		{"TerminateDuration", tr.TerminateDuration == 7*time.Second},
	} {
		if !c.ok {
			t.Errorf("reconnect dropped %s", c.name)
		}
	}
	if again := ct.reconnectTransport().Command; again == got {
		t.Error("two reconnects share one command, want a fresh copy each time")
	}
}

// TestCommandTransportSpecIsIndependent checks that the saved copy of the
// command does not share slices with the caller's command, so editing the
// caller's Args or Env later does not change what a reconnect starts.
func TestCommandTransportSpecIsIndependent(t *testing.T) {
	cmd := exec.Command("server", "--flag")
	cmd.Env = []string{"A=1"}
	ct := newCommandTransport(&mcp.CommandTransport{Command: cmd})

	cmd.Args[1] = "--changed"
	cmd.Env[0] = "A=changed"

	if got, want := ct.spec.Args, []string{"server", "--flag"}; !slices.Equal(got, want) {
		t.Errorf("spec.Args = %v, want %v", got, want)
	}
	if got, want := ct.spec.Env, []string{"A=1"}; !slices.Equal(got, want) {
		t.Errorf("spec.Env = %v, want %v", got, want)
	}
}

// TestCommandTransportHelper is the child process of TestCommandTransportReconnect.
// It writes its working directory to stderr and exits when its stdin closes.
// In a normal test run it does nothing.
func TestCommandTransportHelper(t *testing.T) {
	if os.Getenv("MCPTOOLSET_TEST_HELPER") != "1" {
		t.Skip("helper process for TestCommandTransportReconnect")
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if wd, err = filepath.EvalSymlinks(wd); err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(os.Stderr, "helper started in %s\n", wd)
	_, _ = io.Copy(io.Discard, os.Stdin)
}
