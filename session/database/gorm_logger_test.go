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
	"io"
	"path/filepath"
	"strings"
	"testing"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/session"
)

// dialectorConstructors builds a service through each constructor that opens
// its own *gorm.DB from a dialector and GORM options.
var dialectorConstructors = []struct {
	name string
	open func(d gorm.Dialector, opts ...gorm.Option) (session.Service, error)
}{
	{name: "New", open: func(d gorm.Dialector, opts ...gorm.Option) (session.Service, error) {
		return New(Config{Dialector: d, GORMOptions: opts})
	}},
	{name: "NewSessionService", open: NewSessionService},
}

// openMigrated builds a service on a new SQLite file through open and creates
// its tables.
func openMigrated(t *testing.T, open func(gorm.Dialector, ...gorm.Option) (session.Service, error), opts ...gorm.Option) session.Service {
	t.Helper()
	svc, err := open(sqlite.Open(filepath.Join(t.TempDir(), "sessions.db")), opts...)
	if err != nil {
		t.Fatalf("open() error = %v", err)
	}
	t.Cleanup(func() { _ = svc.(io.Closer).Close() })
	if err := AutoMigrate(svc); err != nil {
		t.Fatalf("AutoMigrate() error = %v", err)
	}
	return svc
}

// New and NewSessionService do not write GORM's log output to the process's
// standard output, which GORM's default logger writes to, as adk-python
// creates its engine with echo=False (database_session_service.py:376-398).
// It replaces logger.Default, so it does not run in parallel.
func TestDialectorConstructors_DiscardGORMLogsByDefault(t *testing.T) {
	var stdout bytes.Buffer
	defaultLogger := logger.Default
	logger.Default = bufferLogger(&stdout)
	t.Cleanup(func() { logger.Default = defaultLogger })
	for _, c := range dialectorConstructors {
		for _, opts := range [][]gorm.Option{nil, {&gorm.Config{}}} {
			stdout.Reset()
			svc := openMigrated(t, c.open, opts...)

			getMissing(t, svc)

			if stdout.Len() != 0 {
				t.Errorf("%s(%d options): GORM's default logger wrote %q, want nothing", c.name, len(opts), stdout.String())
			}
		}
	}
}

// A logger the caller passes in a *gorm.Config receives GORM's log output.
func TestDialectorConstructors_KeepCallerLogger(t *testing.T) {
	for _, c := range dialectorConstructors {
		t.Run(c.name, func(t *testing.T) {
			var logs bytes.Buffer
			svc := openMigrated(t, c.open, &gorm.Config{Logger: bufferLogger(&logs)})

			getMissing(t, svc)

			if !strings.Contains(logs.String(), "record not found") {
				t.Errorf("caller's logger got %q, want a line containing %q", logs.String(), "record not found")
			}
		})
	}
}

// New and NewSessionService leave the caller's options slice as it was, its
// spare capacity included, so a caller can share it between calls.
func TestDialectorConstructors_LeaveCallerOptionsUnchanged(t *testing.T) {
	for _, c := range dialectorConstructors {
		t.Run(c.name, func(t *testing.T) {
			cfg := &gorm.Config{Logger: logger.Discard}
			opts := make([]gorm.Option, 2, 3)
			opts[0], opts[1] = noopOption{}, cfg

			openMigrated(t, c.open, opts...)

			full := opts[:3]
			if full[0] != (noopOption{}) || full[1] != cfg || full[2] != nil {
				t.Errorf("caller's options after the call = %v, want [noopOption, the *gorm.Config, nil]", full)
			}
		})
	}
}
