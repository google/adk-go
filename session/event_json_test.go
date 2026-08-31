// Copyright 2025 Google LLC
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

package session_test

import (
	"context"
	"encoding/json"
	"sort"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func fullyPopulatedEvent() *session.Event {
	return &session.Event{
		LLMResponse: model.LLMResponse{
			Content:      genai.NewContentFromText("hello", genai.RoleModel),
			ModelVersion: "gemini-3-pro",
			Partial:      true,
			TurnComplete: true,
			Interrupted:  true,
			ErrorCode:    "429",
			ErrorMessage: "rate limited",
			AvgLogprobs:  -0.25,
		},
		ID:                 "evt-1",
		Timestamp:          time.Unix(1733040000, 0).UTC(),
		InvocationID:       "inv-1",
		Branch:             "root.planner",
		IsolationScope:     "approval_subflow",
		Author:             "planner",
		LongRunningToolIDs: []string{"tool-1", "tool-2"},
		Routes:             []string{"reviewer", "writer"},
		RequestedInput: &session.RequestInput{
			InterruptID: "approval-1",
			Message:     "Approve?",
			Payload:     map[string]any{"doc": "contract.pdf"},
		},
		Output:   map[string]any{"summary": "ok", "risk_items": float64(3)},
		NodeInfo: &session.NodeInfo{Path: "wf/planner", MessageAsOutput: true, OutputFor: []string{"wf"}},
		Actions: session.EventActions{
			StateDelta:        map[string]any{"stage": "drafted"},
			ArtifactDelta:     map[string]int64{"report.pdf": 2},
			SkipSummarization: true,
			TransferToAgent:   "writer",
			Escalate:          true,
		},
	}
}

// TestEventJSONKeys pins the wire format. Event is persisted whole by session
// services (see session/vertexai raw_event) and read back by other ADK
// runtimes, so its key names are an interface, not an implementation detail.
func TestEventJSONKeys(t *testing.T) {
	b, err := json.Marshal(fullyPopulatedEvent())
	if err != nil {
		t.Fatalf("Marshal() failed: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("Unmarshal() failed: %v", err)
	}
	got := make([]string, 0, len(m))
	for k := range m {
		got = append(got, k)
	}
	sort.Strings(got)

	want := []string{
		"actions", "author", "avgLogprobs", "branch", "content", "errorCode",
		"errorMessage", "id", "interrupted", "invocationId", "isolationScope",
		"longRunningToolIds", "modelVersion", "nodeInfo", "output", "partial",
		"requestedInput", "routes", "timestamp", "turnComplete",
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Event JSON keys mismatch (-want +got):\n%s", diff)
	}
}

// TestEventJSONOmitsEmpty guards against every event carrying a full set of
// nulls, which bloats every persisted row.
func TestEventJSONOmitsEmpty(t *testing.T) {
	b, err := json.Marshal(&session.Event{})
	if err != nil {
		t.Fatalf("Marshal() failed: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("Unmarshal() failed: %v", err)
	}
	got := make([]string, 0, len(m))
	for k := range m {
		got = append(got, k)
	}
	sort.Strings(got)

	want := []string{"actions", "author", "id", "invocationId", "timestamp"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("empty Event JSON keys mismatch (-want +got):\n%s", diff)
	}
}

func TestEventJSONRoundTrip(t *testing.T) {
	want := fullyPopulatedEvent()
	b, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("Marshal() failed: %v", err)
	}
	got := &session.Event{}
	if err := json.Unmarshal(b, got); err != nil {
		t.Fatalf("Unmarshal() failed: %v", err)
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Event round trip mismatch (-want +got):\n%s", diff)
	}
}

// TestEventUnmarshalTimestampDialects covers the three timestamp encodings an
// Event can arrive in: RFC 3339 (this package), epoch seconds as a number
// (adk-python, whose Event.timestamp is a float), and absent.
func TestEventUnmarshalTimestampDialects(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want time.Time
	}{
		{
			name: "rfc3339 string",
			in:   `{"id":"e","timestamp":"2024-12-01T08:00:00Z"}`,
			want: time.Unix(1733040000, 0).UTC(),
		},
		{
			name: "epoch seconds number",
			in:   `{"id":"e","timestamp":1733040000}`,
			want: time.Unix(1733040000, 0).UTC(),
		},
		{
			name: "fractional epoch seconds",
			in:   `{"id":"e","timestamp":1733040000.5}`,
			want: time.Unix(1733040000, 500000000).UTC(),
		},
		{
			name: "millisecond precision",
			in:   `{"id":"e","timestamp":1733040000.123}`,
			want: time.Unix(1733040000, 123000000).UTC(),
		},
		{
			name: "exponent notation",
			in:   `{"id":"e","timestamp":1.7330400e9}`,
			want: time.Unix(1733040000, 0).UTC(),
		},
		{
			name: "zero epoch",
			in:   `{"id":"e","timestamp":0}`,
			want: time.Unix(0, 0).UTC(),
		},
		{
			name: "negative epoch",
			in:   `{"id":"e","timestamp":-86400}`,
			want: time.Unix(-86400, 0).UTC(),
		},
		{
			name: "negative fractional epoch",
			in:   `{"id":"e","timestamp":-0.5}`,
			want: time.Unix(0, -500000000).UTC(),
		},
		{
			name: "absent",
			in:   `{"id":"e"}`,
			want: time.Time{},
		},
		{
			name: "null",
			in:   `{"id":"e","timestamp":null}`,
			want: time.Time{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := &session.Event{}
			if err := json.Unmarshal([]byte(tt.in), got); err != nil {
				t.Fatalf("Unmarshal(%s) failed: %v", tt.in, err)
			}
			if !got.Timestamp.UTC().Equal(tt.want) {
				t.Errorf("Timestamp = %v, want %v", got.Timestamp.UTC(), tt.want)
			}
			if got.ID != "e" {
				t.Errorf("ID = %q, want %q (decoding must not abort early)", got.ID, "e")
			}
		})
	}
}

// TestEventUnmarshalTimestampInvalid covers timestamps that cannot be
// decoded. These must surface an error rather than silently leaving the
// field zero, which would look like a valid epoch.
func TestEventUnmarshalTimestampInvalid(t *testing.T) {
	for _, in := range []string{
		`{"timestamp":"not-a-date"}`,
		`{"timestamp":{"nested":1}}`,
		`{"timestamp":true}`,
		// Out of range for int64 microseconds. Converting these saturates
		// silently, which would yield a plausible-looking instant in the year
		// -290308 and discard the sign, so they must be rejected.
		`{"timestamp":1e300}`,
		`{"timestamp":-1e300}`,
		`{"timestamp":1e19}`,
		`{"timestamp":9223372036854775807}`,
		// Epoch milliseconds read as seconds. Decodes without complaint to the
		// year 56887, which sorts after every real event.
		`{"timestamp":1733040000000}`,
	} {
		got := &session.Event{}
		if err := json.Unmarshal([]byte(in), got); err == nil {
			t.Errorf("Unmarshal(%s) = nil error, want an error", in)
		}
	}
}

// TestEventUnmarshalLegacyFieldNames covers rows written before Event carried
// json tags, when encoding/json used Go field names verbatim. encoding/json
// matches keys case-insensitively, so these must still bind.
func TestEventUnmarshalLegacyFieldNames(t *testing.T) {
	legacy := `{
	  "ID":"evt-1","InvocationID":"inv-1","Author":"planner","Branch":"root",
	  "Timestamp":"2024-12-01T08:00:00Z","Routes":["reviewer"],
	  "Output":{"summary":"ok"},"LongRunningToolIDs":["tool-1"],
	  "Actions":{"StateDelta":{"stage":"drafted"},"TransferToAgent":"writer"},
	  "Partial":true,"ErrorCode":"429"
	}`
	got := &session.Event{}
	if err := json.Unmarshal([]byte(legacy), got); err != nil {
		t.Fatalf("Unmarshal() failed: %v", err)
	}
	want := &session.Event{
		LLMResponse:        model.LLMResponse{Partial: true, ErrorCode: "429"},
		ID:                 "evt-1",
		Timestamp:          time.Unix(1733040000, 0).UTC(),
		InvocationID:       "inv-1",
		Branch:             "root",
		Author:             "planner",
		LongRunningToolIDs: []string{"tool-1"},
		Routes:             []string{"reviewer"},
		Output:             map[string]any{"summary": "ok"},
		Actions: session.EventActions{
			StateDelta:      map[string]any{"stage": "drafted"},
			TransferToAgent: "writer",
		},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("legacy field names mismatch (-want +got):\n%s", diff)
	}
}

// TestEventUnmarshalIntoPopulatedEvent covers decoding into an Event that
// already holds values. encoding/json leaves fields absent from the payload
// alone, and the timestamp retry inside UnmarshalJSON must not change that.
func TestEventUnmarshalIntoPopulatedEvent(t *testing.T) {
	tests := []struct {
		name string
		in   string
	}{
		{"string timestamp", `{"id":"new","timestamp":"2024-12-01T08:00:00Z"}`},
		{"numeric timestamp", `{"id":"new","timestamp":1733040000}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := &session.Event{Branch: "preexisting", Author: "keepme"}
			if err := json.Unmarshal([]byte(tt.in), got); err != nil {
				t.Fatalf("Unmarshal() failed: %v", err)
			}
			want := &session.Event{
				ID:        "new",
				Timestamp: time.Unix(1733040000, 0).UTC(),
				Branch:    "preexisting",
				Author:    "keepme",
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestEventUnmarshalEmptyNodeInfoBecomesNil covers the shape adk-python
// actually emits. Its node_info field is not Optional and it dumps with
// exclude_none, so every event it writes carries a nodeInfo object, including
// events that never came from a workflow. Decoding that to a non-nil pointer
// would contradict the invariant readers rely on, which is that a nil NodeInfo
// means the event is not from a workflow.
func TestEventUnmarshalEmptyNodeInfoBecomesNil(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want *session.NodeInfo
	}{
		{
			name: "absent",
			in:   `{"id":"e"}`,
			want: nil,
		},
		{
			name: "empty object, as adk-python writes for a non-workflow event",
			in:   `{"id":"e","nodeInfo":{}}`,
			want: nil,
		},
		{
			name: "explicitly zero fields",
			in:   `{"id":"e","nodeInfo":{"path":"","messageAsOutput":false,"outputFor":[]}}`,
			want: nil,
		},
		{
			name: "a path alone is real workflow metadata",
			in:   `{"id":"e","nodeInfo":{"path":"wf/planner"}}`,
			want: &session.NodeInfo{Path: "wf/planner"},
		},
		{
			name: "messageAsOutput alone is real workflow metadata",
			in:   `{"id":"e","nodeInfo":{"messageAsOutput":true}}`,
			want: &session.NodeInfo{MessageAsOutput: true},
		},
		{
			name: "outputFor alone is real workflow metadata",
			in:   `{"id":"e","nodeInfo":{"outputFor":["wf"]}}`,
			want: &session.NodeInfo{OutputFor: []string{"wf"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := &session.Event{}
			if err := json.Unmarshal([]byte(tt.in), got); err != nil {
				t.Fatalf("Unmarshal(%s) failed: %v", tt.in, err)
			}
			if diff := cmp.Diff(tt.want, got.NodeInfo); diff != "" {
				t.Errorf("NodeInfo mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestEventJSONPreservesEmptyActionMaps guards the nil-ness of the two map
// fields on EventActions. NewEvent allocates them, and callers write into them
// without a nil check, so a map that survives the trip out to JSON as an empty
// object has to come back as an empty map rather than nil. Tagging these
// omitempty drops the key entirely, and the map then decodes to nil and panics
// on the first write.
func TestEventJSONPreservesEmptyActionMaps(t *testing.T) {
	want := session.NewEvent(context.Background(), "inv-1")
	if want.Actions.StateDelta == nil || want.Actions.ArtifactDelta == nil {
		t.Fatal("NewEvent no longer allocates the action maps; this test is no longer covering anything")
	}

	b, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("Marshal() failed: %v", err)
	}
	got := &session.Event{}
	if err := json.Unmarshal(b, got); err != nil {
		t.Fatalf("Unmarshal() failed: %v", err)
	}

	if got.Actions.StateDelta == nil {
		t.Error("StateDelta is nil after a round trip; writing to it would panic")
	}
	if got.Actions.ArtifactDelta == nil {
		t.Error("ArtifactDelta is nil after a round trip; writing to it would panic")
	}
}

// TestEventActionsJSONDeltaMapEncoding pins how the two delta maps cross the
// wire. A nil map is omitted and an allocated one is written even when empty,
// which is what keeps the nil/empty distinction intact through a round trip
// while never emitting `null`. adk-python rejects `null` for either field --
// both are non-Optional dicts -- and fills an absent key from its default
// factory, so an omitted key is the one encoding both runtimes accept.
func TestEventActionsJSONDeltaMapEncoding(t *testing.T) {
	tests := []struct {
		name    string
		actions session.EventActions
		want    map[string]string // json key -> raw value, "" means absent
	}{
		{
			name:    "nil maps are omitted",
			actions: session.EventActions{},
			want:    map[string]string{"stateDelta": "", "artifactDelta": ""},
		},
		{
			name: "allocated but empty maps are written",
			actions: session.EventActions{
				StateDelta:    map[string]any{},
				ArtifactDelta: map[string]int64{},
			},
			want: map[string]string{"stateDelta": "{}", "artifactDelta": "{}"},
		},
		{
			name: "populated maps are written",
			actions: session.EventActions{
				StateDelta:    map[string]any{"k": "v"},
				ArtifactDelta: map[string]int64{"f.png": 7},
			},
			want: map[string]string{"stateDelta": `{"k":"v"}`, "artifactDelta": `{"f.png":7}`},
		},
		{
			name:    "one allocated, one nil",
			actions: session.EventActions{StateDelta: map[string]any{}},
			want:    map[string]string{"stateDelta": "{}", "artifactDelta": ""},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, err := json.Marshal(&session.Event{ID: "e", Author: "a", Actions: tt.actions})
			if err != nil {
				t.Fatalf("Marshal() failed: %v", err)
			}

			var wire struct {
				Actions map[string]json.RawMessage `json:"actions"`
			}
			if err := json.Unmarshal(b, &wire); err != nil {
				t.Fatalf("Unmarshal() failed: %v", err)
			}
			for key, want := range tt.want {
				raw, ok := wire.Actions[key]
				if want == "" {
					if ok {
						t.Errorf("actions.%s = %s, want the key to be absent", key, raw)
					}
					continue
				}
				if !ok {
					t.Errorf("actions.%s is absent, want %s", key, want)
					continue
				}
				if string(raw) != want {
					t.Errorf("actions.%s = %s, want %s", key, raw, want)
				}
			}

			// Whatever the shape, `null` must never reach adk-python.
			for _, key := range []string{"stateDelta", "artifactDelta"} {
				if string(wire.Actions[key]) == "null" {
					t.Errorf("actions.%s = null; adk-python rejects that for a non-Optional dict", key)
				}
			}

			// And the round trip preserves nil-ness exactly, so a store that
			// persists as JSON hands back the event it was given.
			got := &session.Event{}
			if err := json.Unmarshal(b, got); err != nil {
				t.Fatalf("Unmarshal() failed: %v", err)
			}
			if (got.Actions.StateDelta == nil) != (tt.actions.StateDelta == nil) {
				t.Errorf("StateDelta nil=%v after round trip, want nil=%v",
					got.Actions.StateDelta == nil, tt.actions.StateDelta == nil)
			}
			if (got.Actions.ArtifactDelta == nil) != (tt.actions.ArtifactDelta == nil) {
				t.Errorf("ArtifactDelta nil=%v after round trip, want nil=%v",
					got.Actions.ArtifactDelta == nil, tt.actions.ArtifactDelta == nil)
			}
		})
	}
}
