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

package database

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

// responseFields holds the LLMResponse fields that only the event's own
// columns carry, so a test can compare them without the rest of the event.
type responseFields struct {
	FinishReason            genai.FinishReason
	ModelVersion            string
	AvgLogprobs             float64
	LogprobsResult          *genai.LogprobsResult
	SessionResumptionHandle string
	LiveSessionID           string
}

func responseFieldsOf(ev *session.Event) responseFields {
	return responseFields{
		FinishReason:            ev.FinishReason,
		ModelVersion:            ev.ModelVersion,
		AvgLogprobs:             ev.AvgLogprobs,
		LogprobsResult:          ev.LogprobsResult,
		SessionResumptionHandle: ev.SessionResumptionHandle,
		LiveSessionID:           ev.LiveSessionID,
	}
}

// appendWithResponseFields appends event id to held with every field of
// responseFields set to a non-zero value.
func appendWithResponseFields(t *testing.T, svc session.Service, held session.Session, id string, at time.Time) {
	t.Helper()
	ev := &session.Event{ID: id, InvocationID: "inv1", Author: "model", Timestamp: at}
	ev.FinishReason = genai.FinishReasonStop
	ev.ModelVersion = "gemini-2.5-flash"
	ev.AvgLogprobs = -0.5
	ev.LogprobsResult = &genai.LogprobsResult{
		ChosenCandidates: []*genai.LogprobsResultCandidate{{Token: "hi", LogProbability: -0.25}},
	}
	ev.SessionResumptionHandle = "h1"
	ev.LiveSessionID = "live-sess-db-1"
	if err := svc.AppendEvent(t.Context(), held, ev); err != nil {
		t.Fatalf("AppendEvent(%q) error = %v", id, err)
	}
}

var wantResponseFields = responseFields{
	FinishReason: genai.FinishReasonStop,
	ModelVersion: "gemini-2.5-flash",
	AvgLogprobs:  -0.5,
	LogprobsResult: &genai.LogprobsResult{
		ChosenCandidates: []*genai.LogprobsResultCandidate{{Token: "hi", LogProbability: -0.25}},
	},
	SessionResumptionHandle: "h1",
	LiveSessionID:           "live-sess-db-1",
}

// adk-python's database service stores the whole event
// (schemas/v1.py:222, rebuilt at :236), so the model's finish reason, model
// version, log probabilities, resumption handle and live session id come
// back from Get.
func TestDatabaseService_AppendEvent_ResponseFieldsRoundTrip(t *testing.T) {
	svc := emptyService(t)
	held := createAt(t, svc, "app-fields", "s1", appendT0, nil)
	appendWithResponseFields(t, svc, held, "e1", appendT1)

	got := getStored(t, svc, "app-fields", "s1", 0)

	if got.Events().Len() != 1 {
		t.Fatalf("Get() returned %d events, want 1", got.Events().Len())
	}
	if diff := cmp.Diff(wantResponseFields, responseFieldsOf(got.Events().At(0))); diff != "" {
		t.Errorf("Get() event fields mismatch (-want +got):\n%s", diff)
	}
}

// A row stored before the events table had the response columns reads back
// with those fields empty once AutoMigrate has added the columns, and a new
// append to the same session keeps them.
func TestDatabaseService_AppendEvent_ResponseFieldsAfterMigratingOldTable(t *testing.T) {
	svc := emptyService(t)
	held := createAt(t, svc, "app-fields-old", "s1", appendT0, nil)
	appendWithResponseFields(t, svc, held, "e_old", appendT1)
	for _, column := range []string{"finish_reason", "model_version", "avg_logprobs", "logprobs_result", "session_resumption_handle", "live_session_id"} {
		if err := svc.db.Migrator().DropColumn(&storageEvent{}, column); err != nil {
			t.Fatalf("DropColumn(%q) error = %v", column, err)
		}
	}
	if err := AutoMigrate(svc); err != nil {
		t.Fatalf("AutoMigrate() error = %v", err)
	}
	appendWithResponseFields(t, svc, held, "e_new", appendT2)

	got := getStored(t, svc, "app-fields-old", "s1", 0)

	if got.Events().Len() != 2 {
		t.Fatalf("Get() returned %d events, want 2", got.Events().Len())
	}
	if diff := cmp.Diff(responseFields{}, responseFieldsOf(got.Events().At(0))); diff != "" {
		t.Errorf("Get() fields of the row stored before the migration mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(wantResponseFields, responseFieldsOf(got.Events().At(1))); diff != "" {
		t.Errorf("Get() fields of the row stored after the migration mismatch (-want +got):\n%s", diff)
	}
}

// Events stamped in a zone whose offset changes between them (US Eastern
// clocks went back at 06:00 UTC on 3 November 2024) come back, and are
// filtered and limited, in the order of their instants, as adk-python
// filters and orders by the UTC instant (database_session_service.py:764-778).
// SQLite stores each timestamp as text with its own offset, where "e2"
// (01:00-05:00) sorts before "e1" (01:30-04:00).
func TestDatabaseService_Get_OrdersAndFiltersEventsByInstantAcrossOffsets(t *testing.T) {
	edt := time.FixedZone("EDT", -4*3600)
	est := time.FixedZone("EST", -5*3600)
	tests := []struct {
		name string
		req  session.GetRequest
		want []string
	}{
		{name: "all events", want: []string{"e1", "e2", "e3"}},
		{name: "after", req: session.GetRequest{After: time.Unix(1730613600, 0).UTC()}, want: []string{"e2", "e3"}},
		{name: "after in another zone", req: session.GetRequest{After: time.Unix(1730613600, 0).In(edt)}, want: []string{"e2", "e3"}},
		{name: "num recent events", req: session.GetRequest{NumRecentEvents: 2}, want: []string{"e2", "e3"}},
		{name: "num recent events above the count", req: session.GetRequest{NumRecentEvents: 5}, want: []string{"e1", "e2", "e3"}},
		{name: "after and num recent events", req: session.GetRequest{After: time.Unix(1730611800, 0).UTC(), NumRecentEvents: 1}, want: []string{"e3"}},
	}
	for _, newService := range []struct {
		name string
		new  func(*testing.T) *databaseService
	}{
		{name: "id tiebreak", new: emptyService},
		{name: "append order tiebreak", new: appendOrderService},
	} {
		t.Run(newService.name, func(t *testing.T) {
			svc := newService.new(t)
			held := createAt(t, svc, "app-dst", "s1", time.Unix(1730610000, 0), nil)
			for _, ev := range []struct {
				id string
				at time.Time
			}{
				{"e1", time.Unix(1730611800, 0).In(edt)}, // 01:30 EDT, 05:30 UTC
				{"e2", time.Unix(1730613600, 0).In(est)}, // 01:00 EST, 06:00 UTC
				{"e3", time.Unix(1730615400, 0).In(est)}, // 01:30 EST, 06:30 UTC
			} {
				if err := svc.AppendEvent(t.Context(), held, &session.Event{ID: ev.id, InvocationID: "inv1", Author: "user", Timestamp: ev.at}); err != nil {
					t.Fatalf("AppendEvent(%q) error = %v", ev.id, err)
				}
			}
			for _, tc := range tests {
				t.Run(tc.name, func(t *testing.T) {
					req := tc.req
					req.AppName, req.UserID, req.SessionID = "app-dst", "u1", "s1"

					resp, err := svc.Get(t.Context(), &req)
					if err != nil {
						t.Fatalf("Get() error = %v", err)
					}
					if diff := cmp.Diff(tc.want, eventIDs(resp.Session)); diff != "" {
						t.Errorf("Get() event IDs mismatch (-want +got):\n%s", diff)
					}
				})
			}
		})
	}
}

// Two events at one instant, stamped in different zones, tie and are ordered
// by the service's tiebreak, as adk-python's UTC timestamps tie. As text,
// "e_a" (06:00+00:00) sorts after "e_b" (01:00-05:00), the reverse of both
// the ID order and the append order.
func TestDatabaseService_Get_SameInstantInTwoZonesTies(t *testing.T) {
	instant := time.Unix(1730613600, 0)
	for _, newService := range []struct {
		name string
		new  func(*testing.T) *databaseService
	}{
		{name: "id tiebreak", new: emptyService},
		{name: "append order tiebreak", new: appendOrderService},
	} {
		t.Run(newService.name, func(t *testing.T) {
			svc := newService.new(t)
			held := createAt(t, svc, "app-tie-zones", "s1", time.Unix(1730610000, 0), nil)
			for _, ev := range []struct {
				id string
				at time.Time
			}{
				{"e_a", instant.UTC()},
				{"e_b", instant.In(time.FixedZone("EST", -5*3600))},
			} {
				if err := svc.AppendEvent(t.Context(), held, &session.Event{ID: ev.id, InvocationID: "inv1", Author: "user", Timestamp: ev.at}); err != nil {
					t.Fatalf("AppendEvent(%q) error = %v", ev.id, err)
				}
			}

			got := getStored(t, svc, "app-tie-zones", "s1", 0)

			if diff := cmp.Diff([]string{"e_a", "e_b"}, eventIDs(got)); diff != "" {
				t.Errorf("Get() event IDs mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
