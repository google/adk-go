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

package llminternal

import (
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"syscall"
	"testing"
)

// socketErr rebuilds the error the net package returns for a failed read or
// write on a live socket: *net.OpError wrapping *os.SyscallError wrapping the
// platform errno. The errno value is what differs across platforms, so each
// case passes one explicitly rather than relying on the host's own spelling.
func socketErr(op, syscallName string, errno syscall.Errno) error {
	return &net.OpError{
		Op:     op,
		Net:    "tcp",
		Source: &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 1000},
		Addr:   &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 1001},
		Err:    os.NewSyscallError(syscallName, errno),
	}
}

// Windows sockets report a dropped connection with WSA error numbers, which Go
// does not map onto the POSIX ECONN* constants. Spelled numerically so the test
// compiles and means the same thing on every platform.
const (
	wsaeConnAborted = syscall.Errno(10053)
	wsaeConnReset   = syscall.Errno(10054)
)

func TestIsResumable(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{"nil is not resumable", nil, false},
		{"io.EOF", io.EOF, true},

		// The reader path. The websocket layer turns a dropped connection into
		// a close error whose text carries the code, on every platform.
		{
			name: "reader: abnormal closure 1006",
			err:  errors.New("failed to receive message: websocket: close 1006 (abnormal closure): unexpected EOF"),
			want: true,
		},
		{
			name: "reader: policy violation 1008",
			err:  errors.New("websocket: close 1008 (policy violation)"),
			want: true,
		},
		{"reader: GoAway", errors.New("GoAway received"), true},

		// The sender path. A write straight to the socket surfaces the raw
		// platform error, and its wording is not the reader's. All four of
		// these mean the same thing: the connection this flow was using is
		// gone, so the flow must reconnect rather than give up.
		{
			name: "sender: linux EPIPE",
			err:  socketErr("write", "write", syscall.EPIPE),
			want: true,
		},
		{
			name: "sender: linux ECONNRESET",
			err:  socketErr("write", "write", syscall.ECONNRESET),
			want: true,
		},
		{
			name: "sender: windows WSAECONNABORTED",
			err:  socketErr("write", "wsasend", wsaeConnAborted),
			want: true,
		},
		{
			name: "sender: windows WSAECONNRESET",
			err:  socketErr("write", "wsasend", wsaeConnReset),
			want: true,
		},
		{
			name: "reader: windows WSAECONNRESET on recv",
			err:  socketErr("read", "wsarecv", wsaeConnReset),
			want: true,
		},
		{
			name: "sender: wrapped socket error still resumable",
			err:  fmt.Errorf("sending realtime input: %w", socketErr("write", "wsasend", wsaeConnAborted)),
			want: true,
		},

		// Errors that are about the exchange rather than the transport must
		// still terminate the flow.
		{
			name: "protocol error 1002 is fatal",
			err:  errors.New("websocket: close 1002 (protocol error): fatal"),
			want: false,
		},
		{
			name: "application error is fatal",
			err:  errors.New("model refused the request"),
			want: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isResumable(tc.err); got != tc.want {
				t.Errorf("isResumable(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

// The reader and the sender both report into the same errChan and RunLive acts
// on whichever lands first, so a single dropped connection must get the same
// verdict whichever goroutine saw it. Before the *net.OpError check, the two
// disagreed on Windows: the reader's close text matched "EOF" and resumed,
// while the sender's wsasend error matched nothing and terminated the session.
func TestIsResumableAgreesAcrossReaderAndSender(t *testing.T) {
	platforms := []struct {
		name      string
		readerErr error
		senderErr error
	}{
		{
			name:      "linux",
			readerErr: errors.New("failed to receive message: websocket: close 1006 (abnormal closure): unexpected EOF"),
			senderErr: socketErr("write", "write", syscall.EPIPE),
		},
		{
			name:      "windows",
			readerErr: errors.New("failed to receive message: websocket: close 1006 (abnormal closure): unexpected EOF"),
			senderErr: socketErr("write", "wsasend", wsaeConnAborted),
		},
	}

	for _, p := range platforms {
		t.Run(p.name, func(t *testing.T) {
			reader, sender := isResumable(p.readerErr), isResumable(p.senderErr)
			if reader != sender {
				t.Errorf("same connection loss classified differently: reader=%v sender=%v; "+
					"whether the session resumes would depend on which goroutine reported first",
					reader, sender)
			}
			if !reader {
				t.Errorf("a dropped connection should be resumable, got reader=%v sender=%v", reader, sender)
			}
		})
	}
}
