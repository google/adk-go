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
	"errors"
	"fmt"
	"reflect"
	"slices"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/sessionutils"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
)

// mergeParallelResponses merges the response events of one batch of function
// calls into the batch's single response event. events is in call order; a nil
// entry is a call that gave no response.
//
// It returns nil when no call gave a response, and the event itself when only
// one did. Two or more responses are merged by
// mergeParallelFunctionResponseEvents into a new event with an id from
// platform.NewUUID(ctx), as adk-python builds the merged event anew
// (flows/llm_flows/tools/_batch_executor.py, merge_parallel_function_response_events).
// For each state key that several calls wrote and whose session value is one
// of those writes, the session value is then deep merged over the merged
// delta (_apply_latest_state_writes), so a call that built on another call's
// write keeps its value. ctx may have no session; the merge then follows call
// order alone.
//
// It reads the session state once and fails only when that read fails. The
// flow calls it once per batch, after every call has returned.
func mergeParallelResponses(ctx agent.InvocationContext, events []*session.Event) (*session.Event, error) {
	var state session.ReadonlyState
	if s := ctx.Session(); s != nil {
		state = s.State()
	}
	// Read before the merge, which mutates the first call's delta in place.
	latest, err := latestStateWrites(events, state)
	if err != nil {
		return nil, err
	}
	merged, err := mergeParallelFunctionResponseEvents(events)
	if merged == nil || err != nil {
		return merged, err
	}
	responses := 0
	for _, ev := range events {
		if ev != nil && ev.LLMResponse.Content != nil {
			responses++
		}
	}
	if responses < 2 {
		return merged, nil
	}
	merged.ID = platform.NewUUID(ctx)
	if latest != nil {
		merged.Actions.StateDelta = deepMergeMap(merged.Actions.StateDelta, latest)
	}
	return merged, nil
}

// latestStateWrites returns, for each key that two or more of events wrote
// into their state deltas and whose value in state is one of those writes
// (sameObject), a deep copy of that value. It returns nil when no key
// qualifies or state is nil. A key missing from state is skipped; any other
// read error is returned, without the key, which can carry user data.
//
// The values are copies because the merge deep merges later calls' maps into
// the first call's maps in place, and those are the objects state holds.
func latestStateWrites(events []*session.Event, state session.ReadonlyState) (map[string]any, error) {
	if state == nil {
		return nil, nil
	}
	writes := make(map[string][]any)
	for _, ev := range events {
		if ev == nil {
			continue
		}
		for key, v := range ev.Actions.StateDelta {
			writes[key] = append(writes[key], v)
		}
	}
	var latest map[string]any
	for key, ws := range writes {
		if len(ws) < 2 {
			continue
		}
		v, err := state.Get(key)
		if errors.Is(err, session.ErrStateKeyNotExist) {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("read session state for the merge of parallel calls: %w", err)
		}
		if !slices.ContainsFunc(ws, func(w any) bool { return sameObject(w, v) }) {
			continue
		}
		if latest == nil {
			latest = make(map[string]any)
		}
		latest[key] = sessionutils.CloneValue(v)
	}
	return latest, nil
}

// sameObject reports whether a and b are the same map, or the same slice: same
// type, data address, length and capacity, with a capacity above zero. It is
// the Go form of Python's `is` for dicts and lists. A list emptied in place
// (s[:0]) or filtered into make([]T, 0, n) is the same object as the session's
// value when that is the very slice. A slice with capacity zero never is,
// whatever the pointer says, because zero-capacity slices can share the
// runtime's zero address. A slice of zero-size elements ([]struct{}) never is
// either, empty or not: its array occupies no memory, so the pointer does not
// identify it. Values of any other kind are never the same object.
func sameObject(a, b any) bool {
	if a == nil || b == nil {
		return false
	}
	va, vb := reflect.ValueOf(a), reflect.ValueOf(b)
	if va.Type() != vb.Type() {
		return false
	}
	switch va.Kind() {
	case reflect.Map:
		return va.UnsafePointer() == vb.UnsafePointer()
	case reflect.Slice:
		if va.Type().Elem().Size() == 0 {
			return false
		}
		return va.Cap() > 0 && va.UnsafePointer() == vb.UnsafePointer() && va.Len() == vb.Len() && va.Cap() == vb.Cap()
	default:
		return false
	}
}
