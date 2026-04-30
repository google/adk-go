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

// Package codeexec defines the executor interface that ADK agents use to
// run user-supplied code. Built-in implementations (model-side, container,
// Vertex AI, GKE, agent engine sandbox, unsafe local) live in subpackages
// so their dependencies remain lazy imports.
//
// Mirrors adk-python's google.adk.code_executors. Designed for
// third-party extension: anything that satisfies Executor is a valid
// backend, and the Registry accepts user-defined factories.
package codeexec

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Executor runs a single block of code. Implementations are expected to
// stream partial output via the returned channel; the channel must be
// closed when execution completes.
type Executor interface {
	Name() string

	// Capabilities reports the static feature set of this executor so
	// agents and configuration validators can introspect what's supported
	// without hard-coding provider checks.
	Capabilities() Capabilities

	// Execute runs the supplied input. Returning a non-nil error from
	// Execute itself means the call could not start (e.g. invalid input,
	// runtime not ready); errors that occur during execution are surfaced
	// as Chunk.Err on the final chunk.
	//
	// The returned channel must be closed by the implementation when
	// execution completes. Receivers should drain it until close.
	Execute(ctx context.Context, in Input) (<-chan Chunk, error)
}

// Input carries one execution request.
type Input struct {
	// Language identifies the runtime (e.g. "python", "sql", "bash").
	// Implementations should reject unsupported languages with a clear
	// error.
	Language string

	// Code is the source to execute.
	Code string

	// Stdin is the data piped to the process's standard input, when the
	// executor supports it.
	Stdin []byte

	// Files maps virtual filename → contents. Executors that support file
	// I/O materialize these in the working directory before running.
	Files map[string][]byte

	// SessionID identifies a stateful session; stateful executors persist
	// variables, installed packages, and the working directory across
	// calls with the same SessionID. Empty means a fresh session.
	SessionID string

	// Env are environment variables exposed to the process.
	Env map[string]string

	// TimeoutHint is a soft hint for the desired runtime cap.
	// Implementations may clamp this against their Capabilities.MaxRuntime.
	TimeoutHint time.Duration

	// ProviderOpts is opaque provider-specific configuration (e.g. a
	// GKE namespace, a Vertex AI sandbox image, a Docker resource limit).
	// Each implementation documents the type it accepts.
	ProviderOpts any
}

// Chunk is one streamed output frame. Stdout / Stderr deliver text; Files
// deliver binary outputs the executor produced (e.g. plots, logs).
//
// The final chunk for a successful run sets ExitCode (a pointer so
// callers can distinguish "not yet known" from "0"). The final chunk for
// a failure sets Err in addition to (or instead of) ExitCode.
type Chunk struct {
	Stdout []byte
	Stderr []byte
	Files  map[string][]byte

	// ExitCode, when non-nil, marks this as the final chunk of a run.
	ExitCode *int

	// Err, when non-nil, marks this as the final chunk and indicates the
	// run failed for the documented reason.
	Err error
}

// Capabilities reports a static feature set. Used by callers to validate
// that an executor supports what their agent needs without hard-coding
// provider checks.
type Capabilities struct {
	// Languages is the set of supported language identifiers (case
	// matches what callers pass via Input.Language).
	Languages []string

	// Stateful reports whether SessionID-keyed execution preserves
	// variables / installed packages / working directory across calls.
	Stateful bool

	// NetworkAccess reports whether the executor permits outbound network
	// from running code.
	NetworkAccess bool

	// InstallPackages reports whether running code may install additional
	// packages (e.g. pip install) at runtime.
	InstallPackages bool

	// FileIO reports whether Input.Files / Chunk.Files are honored.
	FileIO bool

	// MaxMemoryBytes is the hard cap on per-execution memory. 0 = no cap.
	MaxMemoryBytes int64

	// MaxRuntime is the hard cap on per-execution wall-clock runtime.
	// 0 = no cap.
	MaxRuntime time.Duration
}

// Factory builds an Executor from a configuration map. Used by the
// registry so apps can configure executors via JSON / YAML.
type Factory func(cfg map[string]any) (Executor, error)

// ErrUnknownExecutor is returned from Lookup when no executor with the
// requested name is registered.
var ErrUnknownExecutor = errors.New("codeexec: unknown executor")

var (
	regMu       sync.RWMutex
	regFactories = map[string]Factory{}
)

// Register adds a factory to the executor registry. Re-registering an
// existing name overwrites the previous entry; this lets test setups
// install fakes safely.
func Register(name string, f Factory) {
	if name == "" {
		panic("codeexec: Register called with empty name")
	}
	if f == nil {
		panic("codeexec: Register called with nil factory")
	}
	regMu.Lock()
	defer regMu.Unlock()
	regFactories[name] = f
}

// Lookup returns the registered factory for name and a boolean
// indicating whether one was found.
func Lookup(name string) (Factory, bool) {
	regMu.RLock()
	defer regMu.RUnlock()
	f, ok := regFactories[name]
	return f, ok
}

// Build constructs an Executor by name. Equivalent to Lookup followed by
// invoking the factory.
func Build(name string, cfg map[string]any) (Executor, error) {
	f, ok := Lookup(name)
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownExecutor, name)
	}
	return f(cfg)
}

// Names returns the sorted list of registered executor names.
func Names() []string {
	regMu.RLock()
	defer regMu.RUnlock()
	out := make([]string, 0, len(regFactories))
	for k := range regFactories {
		out = append(out, k)
	}
	return out
}

// resetRegistry is for tests in this package and its subpackages.
func resetRegistry() {
	regMu.Lock()
	defer regMu.Unlock()
	regFactories = map[string]Factory{}
}
