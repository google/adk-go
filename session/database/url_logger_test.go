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

package database

import (
	"bytes"
	"errors"
	"io"
	"log"
	"strings"
	"testing"

	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/session"
)

// bufferLogger returns a GORM logger that writes everything it logs to buf.
func bufferLogger(buf *bytes.Buffer) logger.Interface {
	return logger.New(log.New(buf, "", 0), logger.Config{LogLevel: logger.Info})
}

// getMissing calls Get for a session that does not exist, which GORM logs as
// "record not found" through the logger it was opened with.
func getMissing(t *testing.T, svc session.Service) {
	t.Helper()
	_, err := svc.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "u1", SessionID: "missing"})
	if !errors.Is(err, session.ErrNotFound) {
		t.Fatalf("Get(missing) error = %v, want an error wrapping session.ErrNotFound", err)
	}
}

// NewSessionServiceFromURL does not write GORM's log output to the process's
// standard output, which GORM's default logger writes to, as adk-python
// creates its engine with echo=False. It replaces logger.Default, so it does
// not run in parallel.
func TestNewSessionServiceFromURL_DiscardsGORMLogsByDefault(t *testing.T) {
	var stdout bytes.Buffer
	defaultLogger := logger.Default
	logger.Default = bufferLogger(&stdout)
	t.Cleanup(func() { logger.Default = defaultLogger })
	for _, opts := range [][]gorm.Option{nil, {&gorm.Config{}}} {
		svc, err := NewSessionServiceFromURL("sqlite:///:memory:", opts...)
		if err != nil {
			t.Fatalf("NewSessionServiceFromURL(%d options) error = %v", len(opts), err)
		}
		t.Cleanup(func() { _ = svc.(io.Closer).Close() })

		getMissing(t, svc)

		if stdout.Len() != 0 {
			t.Errorf("NewSessionServiceFromURL(%d options): GORM's default logger wrote %q, want nothing", len(opts), stdout.String())
		}
	}
}

// A logger the caller passes in a *gorm.Config receives GORM's log output.
func TestNewSessionServiceFromURL_KeepsCallerLogger(t *testing.T) {
	var logs bytes.Buffer
	svc, err := NewSessionServiceFromURL("sqlite:///:memory:", &gorm.Config{Logger: bufferLogger(&logs)})
	if err != nil {
		t.Fatalf("NewSessionServiceFromURL() error = %v", err)
	}
	t.Cleanup(func() { _ = svc.(io.Closer).Close() })

	getMissing(t, svc)

	if !strings.Contains(logs.String(), "record not found") {
		t.Errorf("caller's logger got %q, want a line containing %q", logs.String(), "record not found")
	}
}

// noopOption is a gorm.Option that changes nothing.
type noopOption struct{}

func (noopOption) Apply(*gorm.Config) error       { return nil }
func (noopOption) AfterInitialize(*gorm.DB) error { return nil }

// NewSessionServiceFromURL leaves the caller's options slice as it was, its
// spare capacity included, so a caller can share it between calls.
func TestNewSessionServiceFromURL_LeavesCallerOptionsUnchanged(t *testing.T) {
	cfg := &gorm.Config{Logger: logger.Discard}
	opts := make([]gorm.Option, 2, 3)
	opts[0], opts[1] = noopOption{}, cfg

	svc, err := NewSessionServiceFromURL("sqlite:///:memory:", opts...)
	if err != nil {
		t.Fatalf("NewSessionServiceFromURL() error = %v", err)
	}
	t.Cleanup(func() { _ = svc.(io.Closer).Close() })

	full := opts[:3]
	if full[0] != (noopOption{}) || full[1] != cfg || full[2] != nil {
		t.Errorf("caller's options after the call = %v, want [noopOption, the *gorm.Config, nil]", full)
	}
}
