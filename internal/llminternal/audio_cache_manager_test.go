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
	"bytes"
	"context"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/session"
)

type audioMockArtifacts struct {
	savedName string
	savedPart *genai.Part
}

func (m *audioMockArtifacts) Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	m.savedName = name
	m.savedPart = data
	return &artifact.SaveResponse{Version: 1}, nil
}

func (m *audioMockArtifacts) List(context.Context) (*artifact.ListResponse, error) { return nil, nil }
func (m *audioMockArtifacts) Load(ctx context.Context, name string) (*artifact.LoadResponse, error) {
	return nil, nil
}

func (m *audioMockArtifacts) LoadVersion(ctx context.Context, name string, version int) (*artifact.LoadResponse, error) {
	return nil, nil
}

type audioMockSession struct {
	session.Session
	id      string
	appName string
	userID  string
}

func (m *audioMockSession) ID() string      { return m.id }
func (m *audioMockSession) AppName() string { return m.appName }
func (m *audioMockSession) UserID() string  { return m.userID }

type audioMockInvocationContext struct {
	agent.InvocationContext
	artifacts    agent.Artifacts
	session      session.Session
	invocationID string
	agentObj     agent.Agent
}

func (m *audioMockInvocationContext) Artifacts() agent.Artifacts { return m.artifacts }
func (m *audioMockInvocationContext) Session() session.Session   { return m.session }
func (m *audioMockInvocationContext) InvocationID() string       { return m.invocationID }
func (m *audioMockInvocationContext) Agent() agent.Agent         { return m.agentObj }

func (m *audioMockInvocationContext) Deadline() (time.Time, bool) { return time.Time{}, false }
func (m *audioMockInvocationContext) Done() <-chan struct{}       { return nil }
func (m *audioMockInvocationContext) Err() error                  { return nil }
func (m *audioMockInvocationContext) Value(any) any               { return nil }

type audioMockAgent struct {
	agent.Agent
	name string
}

func (m *audioMockAgent) Name() string { return m.name }

func TestAudioCacheManager(t *testing.T) {
	type chunk struct {
		data []byte
		mime string
	}
	tests := []struct {
		name                string
		inputs              []chunk
		outputs             []chunk
		flushUser           bool
		flushModel          bool
		expectedEventsCount int
		verify              func(t *testing.T, events []*session.Event, mockArt *audioMockArtifacts)
	}{
		{
			name: "FlushBoth_PCM",
			inputs: []chunk{
				{[]byte("input1"), "audio/pcm"},
				{[]byte("input2"), "audio/pcm"},
			},
			outputs: []chunk{
				{[]byte("output1"), "audio/pcm"},
				{[]byte("output2"), "audio/pcm"},
			},
			flushUser:           true,
			flushModel:          true,
			expectedEventsCount: 2,
			verify: func(t *testing.T, events []*session.Event, mockArt *audioMockArtifacts) {
				// Verify input event
				ev1 := events[0]
				if ev1.Author != "user" || ev1.Content.Role != "user" {
					t.Errorf("ev1 author/role mismatch: author=%q, role=%q", ev1.Author, ev1.Content.Role)
				}
				if ev1.Content.Parts[0].FileData.MIMEType != "audio/pcm" {
					t.Errorf("ev1 mimeType mismatch: got %s", ev1.Content.Parts[0].FileData.MIMEType)
				}

				// Verify output event
				ev2 := events[1]
				if ev2.Author != "agent1" || ev2.Content.Role != "model" {
					t.Errorf("ev2 author/role mismatch: author=%q, role=%q", ev2.Author, ev2.Content.Role)
				}
				if ev2.Content.Parts[0].FileData.MIMEType != "audio/pcm" {
					t.Errorf("ev2 mimeType mismatch: got %s", ev2.Content.Parts[0].FileData.MIMEType)
				}
			},
		},
		{
			name: "FlushSelective_InputOnly",
			inputs: []chunk{
				{[]byte("input1"), "audio/pcm"},
			},
			outputs: []chunk{
				{[]byte("output1"), "audio/pcm"},
			},
			flushUser:           true,
			flushModel:          false,
			expectedEventsCount: 1,
			verify: func(t *testing.T, events []*session.Event, mockArt *audioMockArtifacts) {
				if events[0].Author != "user" {
					t.Errorf("Expected author user, got %s", events[0].Author)
				}
			},
		},
		{
			name: "FlushSelective_OutputOnly",
			inputs: []chunk{
				{[]byte("input1"), "audio/pcm"},
			},
			outputs: []chunk{
				{[]byte("output1"), "audio/pcm"},
			},
			flushUser:           false,
			flushModel:          true,
			expectedEventsCount: 1,
			verify: func(t *testing.T, events []*session.Event, mockArt *audioMockArtifacts) {
				if events[0].Author != "agent1" {
					t.Errorf("Expected author agent1, got %s", events[0].Author)
				}
			},
		},
		{
			name:                "FlushEmpty",
			flushUser:           true,
			flushModel:          true,
			expectedEventsCount: 0,
		},
		{
			name: "VerifyCombinedData",
			inputs: []chunk{
				{[]byte("chunk1"), "audio/pcm"},
				{[]byte("chunk2"), "audio/pcm"},
			},
			flushUser:           true,
			flushModel:          false,
			expectedEventsCount: 1,
			verify: func(t *testing.T, events []*session.Event, mockArt *audioMockArtifacts) {
				if mockArt.savedPart == nil {
					t.Fatal("Expected savedPart, got nil")
				}
				expectedData := []byte("chunk1chunk2")
				if !bytes.Equal(mockArt.savedPart.InlineData.Data, expectedData) {
					t.Errorf("Expected combined data %s, got %s", expectedData, mockArt.savedPart.InlineData.Data)
				}
			},
		},
		{
			name: "MimeTypeFallback",
			inputs: []chunk{
				{[]byte("input1"), ""},
			},
			flushUser:           true,
			flushModel:          false,
			expectedEventsCount: 1,
			verify: func(t *testing.T, events []*session.Event, mockArt *audioMockArtifacts) {
				if mockArt.savedPart.InlineData.MIMEType != "audio/pcm" {
					t.Errorf("Expected fallback MIMEType audio/pcm, got %s", mockArt.savedPart.InlineData.MIMEType)
				}
			},
		},
		{
			name: "DifferentMimeTypes",
			inputs: []chunk{
				{[]byte("input1"), "audio/pcm"},
			},
			outputs: []chunk{
				{[]byte("output1"), "audio/mp3"},
			},
			flushUser:           true,
			flushModel:          true,
			expectedEventsCount: 2,
			verify: func(t *testing.T, events []*session.Event, mockArt *audioMockArtifacts) {
				if events[0].Content.Parts[0].FileData.MIMEType != "audio/pcm" {
					t.Errorf("Expected input MIMEType audio/pcm, got %s", events[0].Content.Parts[0].FileData.MIMEType)
				}
				if events[1].Content.Parts[0].FileData.MIMEType != "audio/mp3" {
					t.Errorf("Expected output MIMEType audio/mp3, got %s", events[1].Content.Parts[0].FileData.MIMEType)
				}
			},
		},
		{
			name: "FiltersNonAudio",
			inputs: []chunk{
				{[]byte("input1"), "video/mp4"},
				{[]byte("input2"), "image/png"},
				{[]byte("audio_input"), "audio/pcm"},
			},
			outputs: []chunk{
				{[]byte("output1"), "video/h264"},
			},
			flushUser:           true,
			flushModel:          true,
			expectedEventsCount: 1,
			verify: func(t *testing.T, events []*session.Event, mockArt *audioMockArtifacts) {
				ev := events[0]
				if ev.Author != "user" {
					t.Errorf("Expected author user, got %s", ev.Author)
				}
				if mockArt.savedPart == nil {
					t.Fatal("Expected savedPart, got nil")
				}
				if !bytes.Equal(mockArt.savedPart.InlineData.Data, []byte("audio_input")) {
					t.Errorf("Expected only 'audio_input' to be saved, got %s", mockArt.savedPart.InlineData.Data)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mgr := NewAudioCacheManager()

			for _, in := range tt.inputs {
				mgr.CacheInput(context.Background(), in.data, in.mime)
			}
			for _, out := range tt.outputs {
				mgr.CacheOutput(context.Background(), out.data, out.mime)
			}

			mockArt := &audioMockArtifacts{}
			mockSess := &audioMockSession{id: "sess1", appName: "app1", userID: "user1"}
			mockAg := &audioMockAgent{name: "agent1"}
			mockCtx := &audioMockInvocationContext{
				artifacts:    mockArt,
				session:      mockSess,
				invocationID: "inv1",
				agentObj:     mockAg,
			}

			events, err := mgr.FlushCaches(context.Background(), mockCtx, tt.flushUser, tt.flushModel)
			if err != nil {
				t.Fatalf("FlushCaches failed: %v", err)
			}

			if len(events) != tt.expectedEventsCount {
				t.Fatalf("Expected %d events, got %d", tt.expectedEventsCount, len(events))
			}

			if tt.verify != nil {
				tt.verify(t, events, mockArt)
			}
		})
	}
}
