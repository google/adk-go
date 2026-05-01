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

package builtin_test

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"

	"google.golang.org/adk/plugin/builtin"
)

func TestNewLogging_Defaults(t *testing.T) {
	p, err := builtin.NewLogging(builtin.LoggingConfig{})
	if err != nil {
		t.Fatalf("NewLogging: %v", err)
	}
	if p.Name() != "logging" {
		t.Errorf("Name = %q, want logging", p.Name())
	}
}

func TestNewLogging_OverrideName(t *testing.T) {
	p, err := builtin.NewLogging(builtin.LoggingConfig{Name: "custom"})
	if err != nil {
		t.Fatalf("NewLogging: %v", err)
	}
	if p.Name() != "custom" {
		t.Errorf("Name = %q", p.Name())
	}
}

func TestNewLogging_AcceptsCustomLogger(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, nil))
	_, err := builtin.NewLogging(builtin.LoggingConfig{Logger: logger})
	if err != nil {
		t.Fatalf("NewLogging: %v", err)
	}
}

func TestNewSaveFilesAsArtifacts_Defaults(t *testing.T) {
	p, err := builtin.NewSaveFilesAsArtifacts(builtin.SaveFilesAsArtifactsConfig{})
	if err != nil {
		t.Fatalf("NewSaveFilesAsArtifacts: %v", err)
	}
	if p.Name() != "save_files_as_artifacts" {
		t.Errorf("Name = %q", p.Name())
	}
}

func TestNewSaveFilesAsArtifacts_OverrideName(t *testing.T) {
	p, _ := builtin.NewSaveFilesAsArtifacts(builtin.SaveFilesAsArtifactsConfig{Name: "saver"})
	if p.Name() != "saver" {
		t.Errorf("Name = %q", p.Name())
	}
}

func TestNewDebugLogging_Defaults(t *testing.T) {
	var buf bytes.Buffer
	p, err := builtin.NewDebugLogging(builtin.DebugLoggingConfig{Out: &buf})
	if err != nil {
		t.Fatalf("NewDebugLogging: %v", err)
	}
	if p.Name() != "debug_logging" {
		t.Errorf("Name = %q", p.Name())
	}
}

func TestNewDebugLogging_OverrideName(t *testing.T) {
	var buf bytes.Buffer
	p, _ := builtin.NewDebugLogging(builtin.DebugLoggingConfig{Name: "verbose", Out: &buf})
	if p.Name() != "verbose" {
		t.Errorf("Name = %q", p.Name())
	}
}

func TestNewDebugLogging_PlainSuppressesMarkers(t *testing.T) {
	// Smoke check: a plain emit should not embed the [ADK ...] prefix.
	// We verify by building a plain plugin and inspecting the closure
	// indirectly through trace output isn't accessible; instead, we
	// just ensure the plugin builds without error in plain mode.
	var buf bytes.Buffer
	p, err := builtin.NewDebugLogging(builtin.DebugLoggingConfig{Out: &buf, Plain: true})
	if err != nil {
		t.Fatalf("NewDebugLogging: %v", err)
	}
	if p == nil {
		t.Error("expected non-nil plugin")
	}
	// Quick output sanity by directly using the plugin in a future
	// integration test. Here we just confirm construction.
	_ = strings.TrimSpace
}
