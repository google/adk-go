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

package fileartifact

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

// Delete removes empty directories and Save makes them again. To openChild
// that looks like a directory swapped for something else, which it refuses,
// so it has to tell the two apart.
func TestOpenChildToleratesRecreation(t *testing.T) {
	root := t.TempDir()
	parent, err := os.OpenRoot(root)
	if err != nil {
		t.Fatalf("OpenRoot() failed: %v", err)
	}
	defer func() { _ = parent.Close() }()

	child := filepath.Join(root, "child")
	stop := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			select {
			case <-stop:
				return
			default:
			}
			_ = os.Mkdir(child, 0o700)
			_ = os.Remove(child)
		}
	}()
	defer func() {
		close(stop)
		<-done
	}()

	opened := 0
	for range 20000 {
		dir, err := openChild(parent, "child")
		if err == nil {
			opened++
			_ = dir.Close()
			continue
		}
		if !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("openChild() of a directory that comes and goes: err = %v, want success or fs.ErrNotExist", err)
		}
	}
	if opened == 0 {
		t.Error("openChild() never found the directory")
	}
}

func TestWriteFileKeepsExistingFile(t *testing.T) {
	dir, err := os.OpenRoot(t.TempDir())
	if err != nil {
		t.Fatalf("OpenRoot() failed: %v", err)
	}
	defer func() { _ = dir.Close() }()

	if err := writeFile(dir, "file", []byte("first")); err != nil {
		t.Fatalf("writeFile() failed: %v", err)
	}
	if err := writeFile(dir, "file", []byte("second")); !errors.Is(err, fs.ErrExist) {
		t.Errorf("writeFile() over a file: err = %v, want fs.ErrExist", err)
	}
	got, err := dir.ReadFile("file")
	if err != nil {
		t.Fatalf("ReadFile() failed: %v", err)
	}
	if string(got) != "first" {
		t.Errorf("content after a refused write = %q, want %q", got, "first")
	}
}
