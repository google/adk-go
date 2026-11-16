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
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func TestClosePreventsRetry(t *testing.T) {
	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "1"}, nil)
	connects := 0
	transport := connectFunc(func(ctx context.Context) (mcp.Connection, error) {
		connects++
		ct, st := mcp.NewInMemoryTransports()
		session, err := server.Connect(ctx, st, nil)
		if err != nil {
			return nil, err
		}
		t.Cleanup(func() { _ = session.Close() })
		return ct.Connect(ctx)
	})
	client := newConnectionRefresher(nil, transport)
	t.Cleanup(func() { _ = client.Close() })
	calls := 0
	_, _, err := withRetry(t.Context(), client, func(*mcp.ClientSession) (*mcp.CallToolResult, error) {
		calls++
		// The toolset closes after an operation starts, before its error is handled.
		if err := client.Close(); err != nil {
			t.Fatal(err)
		}
		return nil, mcp.ErrConnectionClosed
	})
	if !errors.Is(err, mcp.ErrConnectionClosed) {
		t.Errorf("withRetry() error = %v, want ErrConnectionClosed", err)
	}
	if calls != 1 || connects != 1 {
		t.Errorf("after Close(): %d calls and %d connections, want 1 each", calls, connects)
	}
}

type connectFunc func(context.Context) (mcp.Connection, error)

func (f connectFunc) Connect(ctx context.Context) (mcp.Connection, error) {
	return f(ctx)
}

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
	defer func() { _ = stderr.Close() }()

	cmd := exec.Command(os.Args[0], "-test.run=^TestCommandTransportHelper$")
	cmd.Env = append(os.Environ(), "MCPTOOLSET_TEST_HELPER=1")
	cmd.Dir = dir
	cmd.Stderr = stderr
	tr := buildTransport(&mcp.CommandTransport{Command: cmd})
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
	extra, err := os.CreateTemp(t.TempDir(), "extra")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = extra.Close() }()
	cmd.ExtraFiles = []*os.File{extra}
	ct := newCommandTransport(&mcp.CommandTransport{Command: cmd})

	cmd.Args[1] = "--changed"
	cmd.Env[0] = "A=changed"
	cmd.ExtraFiles[0] = nil

	if got, want := ct.spec.Args, []string{"server", "--flag"}; !slices.Equal(got, want) {
		t.Errorf("spec.Args = %v, want %v", got, want)
	}
	if got, want := ct.spec.Env, []string{"A=1"}; !slices.Equal(got, want) {
		t.Errorf("spec.Env = %v, want %v", got, want)
	}
	if got, want := ct.spec.ExtraFiles, []*os.File{extra}; !slices.Equal(got, want) {
		t.Errorf("spec.ExtraFiles = %v, want %v", got, want)
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
