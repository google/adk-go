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

type audioMockAgent struct {
	agent.Agent
	name string
}

func (m *audioMockAgent) Name() string { return m.name }

func TestAudioCacheManager_FlushCaches_Both(t *testing.T) {
	mgr := NewAudioCacheManager()

	mgr.CacheInput([]byte("input1"), "audio/pcm")
	mgr.CacheInput([]byte("input2"), "audio/pcm")

	mgr.CacheOutput([]byte("output1"), "audio/pcm")
	mgr.CacheOutput([]byte("output2"), "audio/pcm")

	mockArt := &audioMockArtifacts{}
	mockSess := &audioMockSession{id: "sess1", appName: "app1", userID: "user1"}
	mockAg := &audioMockAgent{name: "agent1"}
	mockCtx := &audioMockInvocationContext{
		artifacts:    mockArt,
		session:      mockSess,
		invocationID: "inv1",
		agentObj:     mockAg,
	}

	events, err := mgr.FlushCaches(mockCtx, true, true)
	if err != nil {
		t.Fatalf("FlushCaches failed: %v", err)
	}

	if len(events) != 2 {
		t.Fatalf("Expected 2 events, got %d", len(events))
	}

	// Verify input event
	ev1 := events[0]
	if ev1.Author != "user" {
		t.Errorf("Expected author user, got %s", ev1.Author)
	}
	if ev1.Content.Role != "user" {
		t.Errorf("Expected role user, got %s", ev1.Content.Role)
	}
	if len(ev1.Content.Parts) != 1 {
		t.Fatalf("Expected 1 part, got %d", len(ev1.Content.Parts))
	}
	p1 := ev1.Content.Parts[0]
	if p1.FileData == nil {
		t.Fatal("Expected FileData, got nil")
	}
	if p1.FileData.MIMEType != "audio/pcm" {
		t.Errorf("Expected MIMEType audio/pcm, got %s", p1.FileData.MIMEType)
	}

	// Verify output event
	ev2 := events[1]
	if ev2.Author != "agent1" {
		t.Errorf("Expected author agent1, got %s", ev2.Author)
	}
	if ev2.Content.Role != "model" {
		t.Errorf("Expected role model, got %s", ev2.Content.Role)
	}
}

func TestAudioCacheManager_FlushCaches_Selective(t *testing.T) {
	mgr := NewAudioCacheManager()

	mgr.CacheInput([]byte("input1"), "audio/pcm")
	mgr.CacheOutput([]byte("output1"), "audio/pcm")

	mockArt := &audioMockArtifacts{}
	mockSess := &audioMockSession{id: "sess1", appName: "app1", userID: "user1"}
	mockAg := &audioMockAgent{name: "agent1"}
	mockCtx := &audioMockInvocationContext{
		artifacts:    mockArt,
		session:      mockSess,
		invocationID: "inv1",
		agentObj:     mockAg,
	}

	// Flush only input
	events, err := mgr.FlushCaches(mockCtx, true, false)
	if err != nil {
		t.Fatalf("FlushCaches failed: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("Expected 1 event, got %d", len(events))
	}
	if events[0].Author != "user" {
		t.Errorf("Expected author user, got %s", events[0].Author)
	}

	// Flush only output
	events, err = mgr.FlushCaches(mockCtx, false, true)
	if err != nil {
		t.Fatalf("FlushCaches failed: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("Expected 1 event, got %d", len(events))
	}
	if events[0].Author != "agent1" {
		t.Errorf("Expected author agent1, got %s", events[0].Author)
	}
}

func TestAudioCacheManager_FlushCaches_Empty(t *testing.T) {
	mgr := NewAudioCacheManager()

	mockArt := &audioMockArtifacts{}
	mockSess := &audioMockSession{id: "sess1", appName: "app1", userID: "user1"}
	mockAg := &audioMockAgent{name: "agent1"}
	mockCtx := &audioMockInvocationContext{
		artifacts:    mockArt,
		session:      mockSess,
		invocationID: "inv1",
		agentObj:     mockAg,
	}

	events, err := mgr.FlushCaches(mockCtx, true, true)
	if err != nil {
		t.Fatalf("FlushCaches failed: %v", err)
	}
	if len(events) != 0 {
		t.Fatalf("Expected 0 events, got %d", len(events))
	}
}

func TestAudioCacheManager_VerifyCombinedData(t *testing.T) {
	mgr := NewAudioCacheManager()

	mgr.CacheInput([]byte("chunk1"), "audio/pcm")
	mgr.CacheInput([]byte("chunk2"), "audio/pcm")

	mockArt := &audioMockArtifacts{}
	mockSess := &audioMockSession{id: "sess1", appName: "app1", userID: "user1"}
	mockAg := &audioMockAgent{name: "agent1"}
	mockCtx := &audioMockInvocationContext{
		artifacts:    mockArt,
		session:      mockSess,
		invocationID: "inv1",
		agentObj:     mockAg,
	}

	_, err := mgr.FlushCaches(mockCtx, true, false)
	if err != nil {
		t.Fatalf("FlushCaches failed: %v", err)
	}

	if mockArt.savedPart == nil {
		t.Fatal("Expected savedPart, got nil")
	}

	expectedData := []byte("chunk1chunk2")
	if !bytes.Equal(mockArt.savedPart.InlineData.Data, expectedData) {
		t.Errorf("Expected combined data %s, got %s", expectedData, mockArt.savedPart.InlineData.Data)
	}
}

func TestAudioCacheManager_MimeTypeFallback(t *testing.T) {
	mgr := NewAudioCacheManager()

	mgr.CacheInput([]byte("input1"), "") // Empty mime type

	mockArt := &audioMockArtifacts{}
	mockSess := &audioMockSession{id: "sess1", appName: "app1", userID: "user1"}
	mockAg := &audioMockAgent{name: "agent1"}
	mockCtx := &audioMockInvocationContext{
		artifacts:    mockArt,
		session:      mockSess,
		invocationID: "inv1",
		agentObj:     mockAg,
	}

	_, err := mgr.FlushCaches(mockCtx, true, false)
	if err != nil {
		t.Fatalf("FlushCaches failed: %v", err)
	}

	if mockArt.savedPart.InlineData.MIMEType != "audio/pcm" {
		t.Errorf("Expected fallback MIMEType audio/pcm, got %s", mockArt.savedPart.InlineData.MIMEType)
	}
}
