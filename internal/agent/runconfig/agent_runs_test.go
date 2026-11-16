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

package runconfig_test

import (
	"fmt"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/internal/agent/runconfig"
)

func TestAgentRuns_LifecycleAndNilSafety(t *testing.T) {
	t.Run("nil receiver", func(t *testing.T) {
		var rc *runconfig.RunConfig
		rc.RecordAgentEnd("a")
		rc.ResetAgentRuns([]string{"a"})
		rc.SeedAgentRuns(map[string]map[string]any{"a": {"k": "v"}}, map[string]bool{"a": true})
		if st, ok := rc.AgentState("a"); st != nil || ok {
			t.Errorf("AgentState(a) = %v, %v; want nil, false", st, ok)
		}
		if rc.AgentEnded("a") {
			t.Error("AgentEnded(a) = true; want false")
		}
	})

	t.Run("zero value records an end in a resumable run", func(t *testing.T) {
		rc := &runconfig.RunConfig{Resumable: true}
		if rc.AgentEnded("a") {
			t.Fatal("AgentEnded(a) before RecordAgentEnd = true; want false")
		}
		rc.RecordAgentEnd("a")
		if !rc.AgentEnded("a") {
			t.Error("AgentEnded(a) after RecordAgentEnd = false; want true")
		}
	})

	t.Run("seeding copies the caller's maps", func(t *testing.T) {
		states := map[string]map[string]any{"a": {"k": "v1"}, "b": {"k": "v2"}}
		ended := map[string]bool{"a": false, "e": true}
		rc := &runconfig.RunConfig{}
		rc.SeedAgentRuns(states, ended)
		states["a"]["k"] = "mutated"
		states["c"] = map[string]any{"k": "v3"}
		ended["a"] = true

		st, ok := rc.AgentState("a")
		if diff := cmp.Diff(map[string]any{"k": "v1"}, st); diff != "" || !ok {
			t.Errorf("AgentState(a) ok = %v, mismatch (-want +got):\n%s", ok, diff)
		}
		if _, ok := rc.AgentState("c"); ok {
			t.Error("AgentState(c) ok = true; want false (added to the caller's map after seeding)")
		}
		if rc.AgentEnded("a") {
			t.Error("AgentEnded(a) = true; want false (seeded false)")
		}
		if !rc.AgentEnded("e") {
			t.Error("AgentEnded(e) = false; want true (seeded true)")
		}
	})

	t.Run("seeding nil maps keeps the store empty", func(t *testing.T) {
		rc := &runconfig.RunConfig{Resumable: true}
		rc.SeedAgentRuns(nil, nil)
		if _, ok := rc.AgentState("a"); ok {
			t.Error("AgentState(a) ok = true; want false")
		}
		rc.RecordAgentEnd("a")
		if !rc.AgentEnded("a") {
			t.Error("AgentEnded(a) after RecordAgentEnd = false; want true")
		}
	})

	t.Run("RecordAgentEnd does nothing in a non-resumable run", func(t *testing.T) {
		rc := &runconfig.RunConfig{Resumable: false}
		rc.SeedAgentRuns(map[string]map[string]any{"a": {"k": "v1"}}, nil)
		rc.RecordAgentEnd("a")
		if rc.AgentEnded("a") {
			t.Error("AgentEnded(a) = true; want false")
		}
		st, ok := rc.AgentState("a")
		if diff := cmp.Diff(map[string]any{"k": "v1"}, st); diff != "" || !ok {
			t.Errorf("AgentState(a) ok = %v, mismatch (-want +got):\n%s", ok, diff)
		}
	})

	t.Run("RecordAgentEnd marks ended and drops the state", func(t *testing.T) {
		rc := &runconfig.RunConfig{Resumable: true}
		rc.SeedAgentRuns(map[string]map[string]any{"a": {"k": "v1"}}, map[string]bool{"a": false})
		rc.RecordAgentEnd("a")
		if !rc.AgentEnded("a") {
			t.Error("AgentEnded(a) = false; want true")
		}
		if st, ok := rc.AgentState("a"); st != nil || ok {
			t.Errorf("AgentState(a) = %v, %v; want nil, false", st, ok)
		}
	})

	t.Run("ResetAgentRuns clears only the named agents", func(t *testing.T) {
		rc := &runconfig.RunConfig{Resumable: true}
		rc.SeedAgentRuns(
			map[string]map[string]any{"a": {"k": "v1"}, "b": {"k": "v2"}},
			map[string]bool{"e": true},
		)
		rc.RecordAgentEnd("x")
		rc.ResetAgentRuns([]string{"a", "e"})

		if _, ok := rc.AgentState("a"); ok {
			t.Error("AgentState(a) ok = true; want false")
		}
		if rc.AgentEnded("e") {
			t.Error("AgentEnded(e) = true; want false")
		}
		st, ok := rc.AgentState("b")
		if diff := cmp.Diff(map[string]any{"k": "v2"}, st); diff != "" || !ok {
			t.Errorf("AgentState(b) ok = %v, mismatch (-want +got):\n%s", ok, diff)
		}
		if !rc.AgentEnded("x") {
			t.Error("AgentEnded(x) = false; want true")
		}
	})
}

func TestAgentRuns_ConcurrentAccess(t *testing.T) {
	const workers = 16
	states := map[string]map[string]any{}
	for i := range workers {
		states[fmt.Sprintf("worker-%d", i)] = map[string]any{"n": i}
		states[fmt.Sprintf("temp-%d", i)] = map[string]any{"n": i}
	}
	rc := &runconfig.RunConfig{Resumable: true}
	rc.SeedAgentRuns(states, nil)

	var wg sync.WaitGroup
	for i := range workers {
		wg.Go(func() {
			worker := fmt.Sprintf("worker-%d", i)
			temp := fmt.Sprintf("temp-%d", i)
			for range 100 {
				rc.AgentState(worker)
				rc.AgentEnded(temp)
				rc.RecordAgentEnd(worker)
				rc.ResetAgentRuns([]string{temp, worker})
			}
			rc.RecordAgentEnd(worker)
		})
	}
	wg.Wait()

	for i := range workers {
		worker := fmt.Sprintf("worker-%d", i)
		if !rc.AgentEnded(worker) {
			t.Errorf("AgentEnded(%s) = false; want true", worker)
		}
		if st, ok := rc.AgentState(worker); st != nil || ok {
			t.Errorf("AgentState(%s) = %v, %v; want nil, false", worker, st, ok)
		}
	}
}
