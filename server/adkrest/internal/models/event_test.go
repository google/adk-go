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

package models_test

import (
	"bytes"
	"encoding/json"
	"math"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/server/adkrest/internal/models"
	"google.golang.org/adk/session"
)

func TestEventRoundTripPreservesTimestamp(t *testing.T) {
	ev := session.Event{
		ID:        "e1",
		Timestamp: time.UnixMicro(1733040000123456).UTC(),
		Author:    "agent",
	}

	restEvent := models.FromSessionEvent(ev)
	if restEvent.Timestamp != 1733040000.123456 {
		t.Errorf("FromSessionEvent().Timestamp = %f, want 1733040000.123456", restEvent.Timestamp)
	}
	rawJSON, err := json.Marshal(restEvent)
	if err != nil {
		t.Fatalf("json.Marshal(restEvent) failed: %v", err)
	}
	var wire map[string]any
	if err := json.Unmarshal(rawJSON, &wire); err != nil {
		t.Fatalf("json.Unmarshal(rawJSON) failed: %v", err)
	}
	if gotTS, ok := wire["timestamp"].(float64); !ok || gotTS != 1733040000.123456 {
		t.Errorf("wire[\"timestamp\"] = %v, want 1733040000.123456", wire["timestamp"])
	}
	var decodedRestEvent models.Event
	if err := json.Unmarshal(rawJSON, &decodedRestEvent); err != nil {
		t.Fatalf("json.Unmarshal(rawJSON, &decodedRestEvent) failed: %v", err)
	}

	back, err := models.ToSessionEvent(decodedRestEvent)
	if err != nil {
		t.Fatalf("ToSessionEvent() failed: %v", err)
	}
	if !back.Timestamp.Equal(ev.Timestamp) {
		t.Errorf("Timestamp = %v, want %v", back.Timestamp, ev.Timestamp)
	}

	zeroBack, err := models.ToSessionEvent(models.FromSessionEvent(session.Event{}))
	if err != nil {
		t.Fatalf("ToSessionEvent(zero event) failed: %v", err)
	}
	if !zeroBack.Timestamp.IsZero() {
		t.Errorf("zero Timestamp round-trip = %v, want zero time.Time", zeroBack.Timestamp)
	}
}

// TestToSessionEventTimestampBounds checks that ToSessionEvent accepts
// timestamps within years 1 to 9999 and rejects out-of-range or NaN values.
func TestToSessionEventTimestampBounds(t *testing.T) {
	tests := []struct {
		name    string
		secs    float64
		want    time.Time
		wantErr bool
	}{
		{name: "sub-second after epoch", secs: 0.5, want: time.UnixMicro(500000).UTC()},
		{name: "sub-second before epoch", secs: -0.5, want: time.UnixMicro(-500000).UTC()},
		{name: "one second", secs: 1, want: time.Unix(1, 0).UTC()},
		{name: "microsecond precision", secs: 1733040000.123456, want: time.UnixMicro(1733040000123456).UTC()},
		{name: "epoch milliseconds", secs: 1733040000123, wantErr: true},
		{name: "overflows int64 positive", secs: 1e13, wantErr: true},
		{name: "overflows int64 negative", secs: -1e13, wantErr: true},
		{name: "huge positive", secs: 1e300, wantErr: true},
		{name: "huge negative", secs: -1e300, wantErr: true},
		{name: "NaN", secs: math.NaN(), wantErr: true},
		{name: "first second of year 1", secs: -62135596800, want: time.Date(1, 1, 1, 0, 0, 0, 0, time.UTC)},
		{name: "before year 1", secs: -62135596801, wantErr: true},
		{name: "last second of year 9999", secs: 253402300799, want: time.Date(9999, 12, 31, 23, 59, 59, 0, time.UTC)},
		{name: "rounds up into year 10000", secs: 253402300799.999999, wantErr: true},
		{name: "first second of year 10000", secs: 253402300800, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := models.ToSessionEvent(models.Event{Timestamp: tt.secs})
			if (err != nil) != tt.wantErr {
				t.Fatalf("ToSessionEvent(%v) error = %v, wantErr %v", tt.secs, err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if !got.Timestamp.Equal(tt.want) {
				t.Errorf("ToSessionEvent(%v).Timestamp = %v, want %v", tt.secs, got.Timestamp, tt.want)
			}
		})
	}
}

// TestMarshalEventDropsNilParts: a nil entry in Content.Parts (a null part in
// a stored create-session body reaches storage verbatim) must serialize, not
// panic every endpoint that returns the session — and the serialized output
// must not contain a null part: REST clients, including the bundled web UI,
// read fields off every part they render. This pins the exact output shape:
// the nil slot is dropped, the real parts keep their order and content.
func TestMarshalEventDropsNilParts(t *testing.T) {
	ev := session.Event{
		ID: "e1",
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{Parts: []*genai.Part{
				genai.NewPartFromText("kept"),
				nil,
				{FunctionCall: &genai.FunctionCall{Name: "f"}},
			}},
		},
	}
	rawJSON, err := json.Marshal(models.FromSessionEvent(ev))
	if err != nil {
		t.Fatalf("json.Marshal(event with nil part) failed: %v", err)
	}
	var got struct {
		Content struct {
			Parts []json.RawMessage `json:"parts"`
		} `json:"content"`
	}
	if err := json.Unmarshal(rawJSON, &got); err != nil {
		t.Fatalf("unmarshal marshaled event: %v", err)
	}
	if len(got.Content.Parts) != 2 {
		t.Fatalf("parts count = %d, want 2 (nil slot dropped): %s", len(got.Content.Parts), rawJSON)
	}
	for i, p := range got.Content.Parts {
		if string(p) == "null" {
			t.Errorf("parts[%d] is null; output must never carry a null part: %s", i, rawJSON)
		}
	}
	if !bytes.Contains(got.Content.Parts[0], []byte(`"kept"`)) {
		t.Errorf("parts[0] = %s, want the kept text part first", got.Content.Parts[0])
	}
	if !bytes.Contains(got.Content.Parts[1], []byte(`"name":"f"`)) || !bytes.Contains(got.Content.Parts[1], []byte(`"args":{}`)) {
		t.Errorf("parts[1] = %s, want the function call with empty args object", got.Content.Parts[1])
	}
}
