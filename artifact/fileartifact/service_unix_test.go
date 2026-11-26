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

//go:build unix

package fileartifact_test

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"syscall"
	"testing"

	"google.golang.org/genai"
)

// A save that fails while creating its directories removes the ones it made.
func TestFailedSaveRemovesItsDirectories(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("permissions do not bind the superuser")
	}
	root := newRoot(t)
	service := newService(t, root)
	// Make the directories of the scope, which stay.
	mustSave(t, service, "other", genai.NewPartFromText("x"))

	// Under this umask a new directory is not writable, so the first one
	// Save makes cannot take the next.
	old := syscall.Umask(0o222)
	_, err := service.Save(t.Context(), saveRequest("dir/file", genai.NewPartFromText("x")))
	syscall.Umask(old)
	if err == nil {
		t.Fatal("Save() into a directory it cannot write to succeeded")
	}
	if got := dirEntries(t, sessionDir(root)); len(got) != 1 || got[0] != "other" {
		t.Errorf("failed Save() left %v, want only [other]", got)
	}
}

// A save that cannot write its files fails, without the path in its error,
// and leaves no version or staging directory behind.
func TestFailedWriteLeavesNoVersion(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("permissions do not bind the superuser")
	}
	root := newRoot(t)
	service := newService(t, root)
	mustSave(t, service, "file", genai.NewPartFromText("x"))

	// Under this umask the staging directory is not writable, so the payload
	// cannot be created in it.
	old := syscall.Umask(0o222)
	_, err := service.Save(t.Context(), saveRequest("file", genai.NewPartFromText("y")))
	syscall.Umask(old)
	if err == nil {
		t.Fatal("Save() into a staging directory it cannot write to succeeded")
	}
	if _, ok := errors.AsType[*fs.PathError](err); ok {
		t.Error("the error carries a path for errors.As")
	}
	versions := filepath.Join(sessionDir(root), "file", "versions")
	if got := dirEntries(t, versions); len(got) != 1 || got[0] != "0" {
		t.Errorf("failed Save() left %v, want only [0]", got)
	}
}
