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
	"context"
	"errors"
	"fmt"
	"sync"

	"google.golang.org/adk/v2/model"
)

const (
	// DefaultModelName is the built-in name of the process-wide default model,
	// as in adk-python (LlmAgent.DEFAULT_MODEL).
	DefaultModelName = "gemini-3.5-flash"
	// DefaultLiveModelName is the built-in name of the process-wide default
	// live model, as in adk-python (LlmAgent.DEFAULT_LIVE_MODEL).
	DefaultLiveModelName = "gemini-live-2.5-flash-native-audio"
)

// defaultModel is one process-wide default: a model object, or a name that is
// resolved through the model registry on first use and kept until a setter
// changes the default.
type defaultModel struct {
	kind string // "default model" or "default live model", for errors

	mu   sync.Mutex // guards the fields below
	obj  model.LLM  // set by setModel; nil when the default is a name
	name string     // set by setName; "" when the default is an object
	memo model.LLM  // the resolved name, nil until resolved
	// gen counts setter calls. A resolution stores its result only when gen
	// has not changed since it read the name, so a setter that runs while
	// the registry factory is building a model always wins.
	gen uint64
}

var (
	defaultNormalModel = &defaultModel{kind: "default model", name: DefaultModelName}
	defaultLiveModel   = &defaultModel{kind: "default live model", name: DefaultLiveModelName}
)

func (d *defaultModel) setModel(m model.LLM) error {
	if m == nil {
		return fmt.Errorf("the %s must not be nil", d.kind)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.obj, d.name, d.memo = m, "", nil
	d.gen++
	return nil
}

func (d *defaultModel) setName(name string) error {
	if name == "" {
		return fmt.Errorf("the %s must be a non-empty name", d.kind)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.obj, d.name, d.memo = nil, name, nil
	d.gen++
	return nil
}

// current returns the default as an object when it is known without
// resolving, or else the name to resolve and the generation it belongs to.
func (d *defaultModel) current() (m model.LLM, name string, gen uint64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.obj != nil {
		return d.obj, "", d.gen
	}
	return d.memo, d.name, d.gen
}

// keep stores m as the resolved default of generation gen and returns the
// model callers of that generation share: m, or the one another caller stored
// first. When a setter has run since gen, m is returned and not stored.
func (d *defaultModel) keep(gen uint64, m model.LLM) model.LLM {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.gen != gen {
		return m
	}
	if d.memo == nil {
		d.memo = m
	}
	return d.memo
}

// resolve returns the default model, building it through model.NewLLM when
// the default is a name that has not been resolved yet. A failed resolution
// is returned and not kept, so a later call tries again.
func (d *defaultModel) resolve(ctx context.Context) (model.LLM, error) {
	m, name, gen := d.current()
	if m != nil {
		return m, nil
	}
	// The model outlives ctx, so a canceled first caller must not cancel
	// what the registry factory builds for every later caller.
	m, err := model.NewLLM(context.WithoutCancel(ctx), name)
	if err != nil {
		return nil, fmt.Errorf("resolve the %s %q: %w", d.kind, name, err)
	}
	if m == nil {
		return nil, fmt.Errorf("resolve the %s %q: %w", d.kind, name, errNilModelFromRegistry)
	}
	return d.keep(gen, m), nil
}

var errNilModelFromRegistry = errors.New("the registered factory returned a nil model")

// SetDefaultModel sets the process-wide default model to m. It applies at
// once to every agent that falls through to the default. It returns an error
// and changes nothing when m is nil. Safe for concurrent use.
func SetDefaultModel(m model.LLM) error { return defaultNormalModel.setModel(m) }

// SetDefaultModelByName sets the process-wide default model to the model the
// registry builds for name, resolved on first use and reused until a setter
// changes the default. It returns an error and changes nothing when name is
// empty. Safe for concurrent use.
func SetDefaultModelByName(name string) error { return defaultNormalModel.setName(name) }

// SetDefaultLiveModel is SetDefaultModel for the default live model.
func SetDefaultLiveModel(m model.LLM) error { return defaultLiveModel.setModel(m) }

// SetDefaultLiveModelByName is SetDefaultModelByName for the default live
// model.
func SetDefaultLiveModelByName(name string) error { return defaultLiveModel.setName(name) }
