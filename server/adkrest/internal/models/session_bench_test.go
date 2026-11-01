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

package models

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/mitchellh/mapstructure"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestFromSessionEquivalence(t *testing.T) {
	for _, n := range []int{0, 1, 10, 100} {
		t.Run(fmt.Sprintf("events=%d", n), func(t *testing.T) {
			sess := seedBenchSession(t, n)
			got, gotErr := FromSession(sess)
			want, wantErr := fromSessionOriginal(sess)
			if (gotErr != nil) != (wantErr != nil) {
				t.Fatalf("FromSession() err = %v, want %v", gotErr, wantErr)
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Fatalf("FromSession() mismatch (-want +got):\n%s", diff)
			}
			gotBytes, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			wantBytes, err := json.Marshal(want)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(gotBytes, wantBytes) {
				t.Fatalf("JSON bytes mismatch:\ngot:  %s\nwant: %s", gotBytes, wantBytes)
			}
		})
	}
}

func TestSessionIDFromHTTPParametersEquivalence(t *testing.T) {
	cases := []struct {
		name string
		vars map[string]string
	}{
		{
			name: "all parameters present",
			vars: map[string]string{"app_name": "app1", "user_id": "u1", "session_id": "s1"},
		},
		{
			name: "optional session_id omitted",
			vars: map[string]string{"app_name": "app1", "user_id": "u1"},
		},
		{
			name: "missing app_name",
			vars: map[string]string{"user_id": "u1", "session_id": "s1"},
		},
		{
			name: "missing user_id",
			vars: map[string]string{"app_name": "app1", "session_id": "s1"},
		},
		{
			name: "empty map",
			vars: map[string]string{},
		},
		{
			name: "nil map",
			vars: nil,
		},
		{
			name: "extra route vars ignored",
			vars: map[string]string{"app_name": "app1", "user_id": "u1", "session_id": "s1", "version": "3"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, gotErr := SessionIDFromHTTPParameters(tc.vars)
			want, wantErr := sessionIDFromHTTPParametersOriginal(tc.vars)
			if (gotErr != nil) != (wantErr != nil) {
				t.Fatalf("err = %v, want %v", gotErr, wantErr)
			}
			if gotErr != nil && gotErr.Error() != wantErr.Error() {
				t.Fatalf("err = %q, want %q", gotErr.Error(), wantErr.Error())
			}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Fatalf("SessionID mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func BenchmarkFromSession(b *testing.B) {
	for _, n := range []int{100, 1000} {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			sess := seedBenchSession(b, n)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				got, err := FromSession(sess)
				if err != nil {
					b.Fatal(err)
				}
				if len(got.Events) != n {
					b.Fatalf("len(Events) = %d, want %d", len(got.Events), n)
				}
			}
		})
	}
}

func BenchmarkSessionIDFromHTTPParameters(b *testing.B) {
	vars := map[string]string{
		"app_name":   "my-app",
		"user_id":    "user-123",
		"session_id": "sess-456",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, err := SessionIDFromHTTPParameters(vars)
		if err != nil {
			b.Fatal(err)
		}
		if got.ID != "sess-456" {
			b.Fatalf("got.ID = %q", got.ID)
		}
	}
}

func fromSessionOriginal(session session.Session) (Session, error) {
	state := map[string]any{}
	maps.Insert(state, session.State().All())
	events := []Event{}
	for event := range session.Events().All() {
		events = append(events, FromSessionEvent(*event))
	}
	mappedSession := Session{
		ID:        session.ID(),
		AppName:   session.AppName(),
		UserID:    session.UserID(),
		UpdatedAt: session.LastUpdateTime().Unix(),
		Events:    events,
		State:     state,
	}
	return mappedSession, mappedSession.Validate()
}

func sessionIDFromHTTPParametersOriginal(vars map[string]string) (SessionID, error) {
	var sessionID SessionID
	decoder, err := mapstructure.NewDecoder(&mapstructure.DecoderConfig{
		WeaklyTypedInput: true,
		Result:           &sessionID,
	})
	if err != nil {
		return sessionID, err
	}
	err = decoder.Decode(vars)
	if err != nil {
		return sessionID, err
	}
	if sessionID.AppName == "" {
		return sessionID, fmt.Errorf("app_name parameter is required")
	}
	if sessionID.UserID == "" {
		return sessionID, fmt.Errorf("user_id parameter is required")
	}
	return sessionID, nil
}

var benchText = strings.Repeat("The quick brown fox jumps over the lazy dog. ", 7)

func seedBenchSession(tb testing.TB, n int) session.Session {
	tb.Helper()
	ctx := tb.Context()
	svc := session.InMemoryService()
	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   "app",
		UserID:    "user",
		SessionID: "sess",
		State:     map[string]any{"k1": "v1", "k2": 42},
	})
	if err != nil {
		tb.Fatal(err)
	}
	start := time.Unix(1_700_000_000, 0)
	for i := 0; i < n; i++ {
		ev := &session.Event{
			ID:           fmt.Sprintf("ev-%06d", i),
			InvocationID: fmt.Sprintf("inv-%06d", i/2),
			Timestamp:    start.Add(time.Duration(i) * time.Second),
		}
		switch {
		case i%2 == 0:
			ev.Author = "user"
			ev.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText(benchText, genai.RoleUser)}
		case i%8 == 1:
			ev.Author = "agent"
			ev.LLMResponse = model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{
				FunctionCall: &genai.FunctionCall{ID: fmt.Sprintf("call-%06d", i), Name: "lookup", Args: map[string]any{"q": "weather"}},
			}}}}
		case i%8 == 3:
			ev.Author = "agent"
			ev.LLMResponse = model.LLMResponse{Content: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{ID: fmt.Sprintf("call-%06d", i-2), Name: "lookup", Response: map[string]any{"result": benchText}},
			}}}}
		default:
			ev.Author = "agent"
			ev.LLMResponse = model.LLMResponse{Content: genai.NewContentFromText(benchText, genai.RoleModel)}
		}
		if i%10 == 9 {
			ev.Actions.StateDelta = map[string]any{"turn": i}
		}
		if err := svc.AppendEvent(ctx, created.Session, ev); err != nil {
			tb.Fatal(err)
		}
	}
	resp, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "sess"})
	if err != nil {
		tb.Fatal(err)
	}
	return resp.Session
}
