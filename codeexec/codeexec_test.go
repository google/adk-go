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

package codeexec

import (
	"context"
	"errors"
	"testing"
)

type fakeExec struct{}

func (f *fakeExec) Name() string                 { return "fake" }
func (f *fakeExec) Capabilities() Capabilities   { return Capabilities{Languages: []string{"go"}} }
func (f *fakeExec) Execute(_ context.Context, _ Input) (<-chan Chunk, error) {
	c := make(chan Chunk, 1)
	exit := 0
	c <- Chunk{Stdout: []byte("ok"), ExitCode: &exit}
	close(c)
	return c, nil
}

func TestRegisterAndLookup(t *testing.T) {
	resetRegistry()
	Register("fake", func(_ map[string]any) (Executor, error) { return &fakeExec{}, nil })
	f, ok := Lookup("fake")
	if !ok {
		t.Fatal("Lookup should find registered factory")
	}
	exec, err := f(nil)
	if err != nil {
		t.Fatalf("factory: %v", err)
	}
	if exec.Name() != "fake" {
		t.Errorf("Name = %q", exec.Name())
	}
}

func TestBuild_UnknownReturnsError(t *testing.T) {
	resetRegistry()
	_, err := Build("does-not-exist", nil)
	if !errors.Is(err, ErrUnknownExecutor) {
		t.Errorf("err = %v, want ErrUnknownExecutor", err)
	}
}

func TestNames(t *testing.T) {
	resetRegistry()
	Register("a", func(_ map[string]any) (Executor, error) { return &fakeExec{}, nil })
	Register("b", func(_ map[string]any) (Executor, error) { return &fakeExec{}, nil })
	names := Names()
	if len(names) != 2 {
		t.Errorf("Names = %v, want 2 entries", names)
	}
}

func TestRegister_PanicsOnEmptyName(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected panic")
		}
	}()
	Register("", func(_ map[string]any) (Executor, error) { return nil, nil })
}

func TestRegister_PanicsOnNilFactory(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected panic")
		}
	}()
	Register("x", nil)
}

func TestExecute_Basic(t *testing.T) {
	exec := &fakeExec{}
	ch, err := exec.Execute(context.Background(), Input{Language: "go", Code: "x"})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	var got []Chunk
	for c := range ch {
		got = append(got, c)
	}
	if len(got) != 1 {
		t.Fatalf("chunks = %d, want 1", len(got))
	}
	if string(got[0].Stdout) != "ok" {
		t.Errorf("stdout = %q", got[0].Stdout)
	}
	if got[0].ExitCode == nil || *got[0].ExitCode != 0 {
		t.Errorf("ExitCode = %v, want 0", got[0].ExitCode)
	}
}
