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

package workflow

import (
	"iter"
	"maps"
	"sync"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// SchemaState is an in-memory session.State whose writes are checked against a state
// schema before they are stored. It is safe for concurrent use.
//
// It holds two maps, as adk-python's State does: the committed value it was
// started from, and the pending delta that writes go to. A read sees the delta
// first. Every write is checked in full before anything is stored, so a refused
// write leaves the state as it was. A key that contains a colon is never
// checked (see [StateSchemaError]).
//
// The zero SchemaState is an empty state with no schema.
type SchemaState struct {
	validator agent.StateValidator // nil: nothing is checked
	value     map[string]any       // never written after construction

	mu    sync.RWMutex
	delta map[string]any // guarded by mu; made on first write
}

// SchemaStateConfig configures NewSchemaState.
type SchemaStateConfig struct {
	// Schema is the state schema. Nil means no schema: nothing is checked. A zero
	// *jsonschema.Resolved (one with no schema document) is an empty schema: it
	// declares no key, so every key without a colon is refused.
	Schema *jsonschema.Resolved
	// Value is the committed state the new SchemaState starts from. The map is copied
	// (its values are shared with the caller) and is not checked against Schema.
	// Nil means empty.
	Value map[string]any
	// Delta is the pending delta the new SchemaState starts from. The map is copied
	// (its values are shared with the caller) and is not checked against Schema.
	// Nil means empty.
	Delta map[string]any
}

// NewSchemaState returns a SchemaState over copies of the maps cfg.Value and cfg.Delta.
//
// cfg.Schema must not refer to another document; it is never modified. When
// it does refer to one, NewSchemaState returns nil and an error that matches
// [ErrStateSchemaUnusable]. A nil cfg.Schema is not an error: the state then
// accepts every key and value.
func NewSchemaState(cfg SchemaStateConfig) (*SchemaState, error) {
	s := &SchemaState{value: maps.Clone(cfg.Value), delta: maps.Clone(cfg.Delta)}
	if cfg.Schema == nil {
		return s, nil
	}
	v, err := newStateValidator(cfg.Schema)
	if err != nil {
		return nil, err
	}
	s.validator = v
	return s, nil
}

var _ session.State = (*SchemaState)(nil)

// Get returns the value of key in the pending delta, else in the committed
// value. It returns nil and [session.ErrStateKeyNotExist] when neither holds
// key. A key holding nil is present: Get returns nil, nil. Reads are not
// checked against the schema.
func (s *SchemaState) Get(key string) (any, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if v, ok := s.lookup(key); ok {
		return v, nil
	}
	return nil, session.ErrStateKeyNotExist
}

// Set checks the entry against the schema and, when it is accepted, stores
// value in the pending delta. value is stored as given, not copied or
// converted. A refused entry returns a *[StateSchemaError] and changes
// nothing.
func (s *SchemaState) Set(key string, value any) error {
	if s.validator != nil {
		if err := s.validator.ValidateStateEntry(key, value); err != nil {
			return err
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.store(key, value)
	return nil
}

// All yields a snapshot, taken when All is called, of the committed value
// overlaid with the pending delta: a delta entry wins. The order is not
// defined. No lock is held while the caller ranges, so the loop body may write
// to s; the writes do not show in the loop.
func (s *SchemaState) All() iter.Seq2[string, any] {
	s.mu.RLock()
	snap := make(map[string]any, len(s.value)+len(s.delta))
	maps.Copy(snap, s.value)
	maps.Copy(snap, s.delta)
	s.mu.RUnlock()
	return maps.All(snap)
}

// Update checks every entry of entries against the schema and, only when all
// are accepted, stores them all in the pending delta in one step: a concurrent
// reader sees all of them or none. When an entry is refused, Update returns
// the *[StateSchemaError] of the first refused entry in sorted key order and
// stores nothing, not even the entries that would have been accepted. A nil or
// empty entries changes nothing.
//
// A node's state has no Update. A node writes in bulk by yielding an event
// whose Actions.StateDelta holds the entries, which are checked together
// before the event is sent; the node itself reads them only after the event
// is appended.
func (s *SchemaState) Update(entries map[string]any) error {
	if err := validateEntries(s.validator, entries); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for k, v := range entries {
		s.store(k, v)
	}
	return nil
}

// SetDefault returns the value of key when s holds it, and otherwise stores def
// under key and returns def. It looks in the pending delta first, then in the
// committed value; a key holding nil is present. When key is present, def is
// neither checked nor stored. When key is absent, def is checked as [Set]
// checks a value; a refused def returns nil and the *[StateSchemaError] and
// changes nothing, so SetDefault(key, nil) for a string property is refused.
//
// SetDefault is atomic: the lookup and the write happen under one lock, so of
// several concurrent calls for one absent key, one stores its default and all
// return it.
//
// # In a node
//
// A node's state has no SetDefault. A node calls Get and then Set when Get
// returns [session.ErrStateKeyNotExist]. That pair is not atomic against
// another writer of the same key running at the same time, and that includes a
// sibling node of a fan-out: sibling nodes run on their own goroutines over one
// session state, so both may miss the key and each store its own default.
// adk-python's State.setdefault is atomic across nodes.
func (s *SchemaState) SetDefault(key string, def any) (any, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if v, ok := s.lookup(key); ok {
		return v, nil
	}
	if s.validator != nil {
		if err := s.validator.ValidateStateEntry(key, def); err != nil {
			return nil, err
		}
	}
	s.store(key, def)
	return def, nil
}

// Delta returns a copy of the pending delta: every entry of cfg.Delta and
// every entry written since, with the latest value per key. The caller owns
// the map, which is never nil; an empty map means nothing is pending.
func (s *SchemaState) Delta() map[string]any {
	s.mu.RLock()
	defer s.mu.RUnlock()
	d := make(map[string]any, len(s.delta))
	maps.Copy(d, s.delta)
	return d
}

// lookup returns the value of key in the delta, else in the value. The caller
// holds mu.
func (s *SchemaState) lookup(key string) (any, bool) {
	if v, ok := s.delta[key]; ok {
		return v, true
	}
	v, ok := s.value[key]
	return v, ok
}

// store writes key to the delta. The caller holds mu for writing.
func (s *SchemaState) store(key string, value any) {
	if s.delta == nil {
		s.delta = map[string]any{}
	}
	s.delta[key] = value
}
