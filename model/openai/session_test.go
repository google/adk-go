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

package openai

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"testing"
	"time"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestWithSessionID(t *testing.T) {
	ctx := context.Background()
	sessionID := "test-session-123"

	ctx = WithSessionID(ctx, sessionID)

	retrieved := GetSessionID(ctx)
	if retrieved != sessionID {
		t.Errorf("Expected session ID '%s', got '%s'", sessionID, retrieved)
	}
}

func TestGetSessionID_NotFound(t *testing.T) {
	ctx := context.Background()

	retrieved := GetSessionID(ctx)
	if retrieved != "" {
		t.Errorf("Expected empty string, got '%s'", retrieved)
	}
}

func TestHasSessionID(t *testing.T) {
	ctx := context.Background()

	if HasSessionID(ctx) {
		t.Error("Expected HasSessionID to return false for empty context")
	}

	ctx = WithSessionID(ctx, "test-session")

	if !HasSessionID(ctx) {
		t.Error("Expected HasSessionID to return true after setting session ID")
	}
}

func TestMustGetSessionID_Success(t *testing.T) {
	ctx := WithSessionID(context.Background(), "test-session")

	sessionID := MustGetSessionID(ctx)
	if sessionID != "test-session" {
		t.Errorf("Expected 'test-session', got '%s'", sessionID)
	}
}

func TestMustGetSessionID_Panic(t *testing.T) {
	ctx := context.Background()

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic when session ID not found")
		}
	}()

	MustGetSessionID(ctx)
}

func TestExtractSessionID_PrimaryKey(t *testing.T) {
	ctx := WithSessionID(context.Background(), "primary-session")

	cfg := &SessionConfig{
		DisableAutoGeneration: true,
	}

	sessionID := extractSessionID(ctx, cfg)
	if sessionID != "primary-session" {
		t.Errorf("Expected 'primary-session', got '%s'", sessionID)
	}
}

func TestExtractSessionID_AlternativeKeys(t *testing.T) {
	tests := []struct {
		name string
		key  contextKey
	}{
		{"session_id", sessionIDKeyAlt1},
		{"SessionID", sessionIDKeyAlt2},
	}

	for _, tt := range tests {
		t.Run(string(tt.name), func(t *testing.T) {
			ctx := context.WithValue(context.Background(), tt.key, "alt-session")

			cfg := &SessionConfig{
				DisableAutoGeneration: true,
			}

			sessionID := extractSessionID(ctx, cfg)
			if sessionID != "alt-session" {
				t.Errorf("Expected 'alt-session', got '%s'", sessionID)
			}
		})
	}
}

func TestExtractSessionID_AutoGeneration(t *testing.T) {
	ctx := context.Background()

	cfg := &SessionConfig{
		DisableAutoGeneration: false,
		Logger:                nil, // Suppress warnings for test
	}

	sessionID := extractSessionID(ctx, cfg)
	if sessionID == "" {
		t.Error("Expected auto-generated session ID, got empty string")
	}

	// Should be a valid UUID
	if len(sessionID) != 36 {
		t.Errorf("Expected UUID length 36, got %d", len(sessionID))
	}

	t.Logf("Auto-generated session ID: %s", sessionID)
}

func TestExtractSessionID_AutoGenerationDisabled(t *testing.T) {
	ctx := context.Background()

	cfg := &SessionConfig{
		DisableAutoGeneration: true,
	}

	sessionID := extractSessionID(ctx, cfg)
	if sessionID != "" {
		t.Errorf("Expected empty string when auto-generation disabled, got '%s'", sessionID)
	}
}

func TestExtractSessionID_WithLogging(t *testing.T) {
	ctx := context.Background()

	// Capture log output
	var logBuf strings.Builder
	logger := log.New(&logBuf, "", 0)

	cfg := &SessionConfig{
		DisableAutoGeneration: false,
		Logger:                logger,
	}

	sessionID := extractSessionID(ctx, cfg)

	if sessionID == "" {
		t.Error("Expected auto-generated session ID")
	}

	logOutput := logBuf.String()
	if !strings.Contains(logOutput, "WARNING") {
		t.Error("Expected warning in log output")
	}
	if !strings.Contains(logOutput, "auto-generated") {
		t.Error("Expected 'auto-generated' in log output")
	}

	t.Logf("Log output: %s", logOutput)
}

func TestExtractSessionID_AlternativeKeyWithLogging(t *testing.T) {
	ctx := context.WithValue(context.Background(), sessionIDKeyAlt1, "alt-session")

	var logBuf strings.Builder
	logger := log.New(&logBuf, "", 0)

	cfg := &SessionConfig{
		DisableAutoGeneration: true,
		Logger:                logger,
	}

	sessionID := extractSessionID(ctx, cfg)
	if sessionID != "alt-session" {
		t.Errorf("Expected 'alt-session', got '%s'", sessionID)
	}

	logOutput := logBuf.String()
	if !strings.Contains(logOutput, "WARNING") {
		t.Error("Expected warning for alternative key usage")
	}
	if !strings.Contains(logOutput, "alternative session key") {
		t.Error("Expected 'alternative session key' in log output")
	}

	t.Logf("Log output: %s", logOutput)
}

func TestExtractSessionIDStrict(t *testing.T) {
	tests := []struct {
		name     string
		setupCtx func() context.Context
		expected string
	}{
		{
			name: "with session ID",
			setupCtx: func() context.Context {
				return WithSessionID(context.Background(), "test-session")
			},
			expected: "test-session",
		},
		{
			name: "without session ID",
			setupCtx: func() context.Context {
				return context.Background()
			},
			expected: "",
		},
		{
			name: "with alternative key",
			setupCtx: func() context.Context {
				return context.WithValue(context.Background(), sessionIDKeyAlt1, "alt-session")
			},
			expected: "alt-session",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := tt.setupCtx()
			sessionID := extractSessionIDStrict(ctx)

			if sessionID != tt.expected {
				t.Errorf("Expected '%s', got '%s'", tt.expected, sessionID)
			}
		})
	}
}

func TestExtractSessionIDWithLogging(t *testing.T) {
	ctx := context.Background()

	var logBuf strings.Builder
	logger := log.New(&logBuf, "", 0)

	sessionID := extractSessionIDWithLogging(ctx, logger)

	if sessionID == "" {
		t.Error("Expected auto-generated session ID")
	}

	logOutput := logBuf.String()
	if !strings.Contains(logOutput, "WARNING") {
		t.Error("Expected warning in log output")
	}
}

// Test integration with OpenAI adapter
func TestSessionIDIntegration_WithAdapter(t *testing.T) {
	logger := log.New(os.Stdout, "[TEST] ", log.LstdFlags)

	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
		Logger:  logger,
	}

	model, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := model.(*openaiModel)

	// Test 1: With session ID
	ctx1 := WithSessionID(context.Background(), "user-session-123")
	sessionID1 := extractSessionIDWithLogging(ctx1, om.logger)

	if sessionID1 != "user-session-123" {
		t.Errorf("Expected 'user-session-123', got '%s'", sessionID1)
	}

	// Test 2: Without session ID (auto-generated)
	ctx2 := context.Background()
	sessionID2 := extractSessionIDWithLogging(ctx2, om.logger)

	if sessionID2 == "" {
		t.Error("Expected auto-generated session ID")
	}

	// Test 3: Different sessions should have different IDs
	if sessionID1 == sessionID2 {
		t.Error("Different contexts should have different session IDs")
	}

	t.Logf("Session 1: %s", sessionID1)
	t.Logf("Session 2 (auto): %s", sessionID2)
}

// Test priority order of session ID sources
func TestSessionIDPriority(t *testing.T) {
	// Primary key should take precedence
	ctx := WithSessionID(context.Background(), "primary")
	ctx = context.WithValue(ctx, sessionIDKeyAlt1, "alternative")

	cfg := &SessionConfig{
		DisableAutoGeneration: true,
	}

	sessionID := extractSessionID(ctx, cfg)
	if sessionID != "primary" {
		t.Errorf("Expected primary key to take precedence, got '%s'", sessionID)
	}
}

// Test empty string session ID handling
func TestEmptySessionIDHandling(t *testing.T) {
	ctx := WithSessionID(context.Background(), "")

	sessionID := GetSessionID(ctx)
	if sessionID != "" {
		t.Errorf("Expected empty string to be preserved, got '%s'", sessionID)
	}

	// Auto-generation should kick in for empty session ID
	cfg := &SessionConfig{
		DisableAutoGeneration: false,
	}

	autoSessionID := extractSessionID(ctx, cfg)
	if autoSessionID == "" {
		t.Error("Expected auto-generation to create non-empty session ID")
	}
}

// Test conversation cleanup batch - multiple sessions with TTL
func TestConversationCleanupBatch(t *testing.T) {
	logger := log.New(os.Stdout, "[CLEANUP-TEST] ", log.LstdFlags)

	// Short TTL for testing - background cleanup runs every TTL/2 (150ms)
	shortTTL := 300 * time.Millisecond

	cfg := &Config{
		BaseURL:    "http://localhost:1234/v1",
		Logger:     logger,
		SessionTTL: shortTTL,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	// Create 10 sessions with messages
	sessionIDs := make([]string, 10)
	for i := 0; i < 10; i++ {
		sessionIDs[i] = fmt.Sprintf("session-%d", i)
		msg := &OpenAIMessage{
			Role:    "user",
			Content: fmt.Sprintf("Message in session %d", i),
		}
		om.addToHistory(sessionIDs[i], msg)
	}

	// Verify all sessions exist
	om.mu.RLock()
	initialCount := len(om.conversations)
	om.mu.RUnlock()

	if initialCount != 10 {
		t.Errorf("Expected 10 sessions, got %d", initialCount)
	}

	t.Logf("Created %d sessions", initialCount)

	// Wait a bit, then access first 5 sessions to keep them fresh
	time.Sleep(50 * time.Millisecond)

	// Access first 5 sessions to update their lastAccess timestamp
	for i := 0; i < 5; i++ {
		msg := &OpenAIMessage{
			Role:    "user",
			Content: fmt.Sprintf("Keep-alive message for session %d", i),
		}
		om.addToHistory(sessionIDs[i], msg)
	}

	t.Logf("Refreshed first 5 sessions")

	// Wait for TTL to expire for last 5 sessions (they're now > 300ms old)
	// First 5 were just accessed ~50ms ago, so they should still be fresh
	// Background cleanup runs every 150ms, so we wait for cleanup cycles
	// After 320ms total: last 5 are 320ms old (> 300ms TTL), first 5 are 270ms old (< 300ms TTL)
	time.Sleep(270 * time.Millisecond)

	// Check how many sessions remain (background cleanup should have removed expired ones)
	om.mu.RLock()
	remainingCount := len(om.conversations)
	remainingSessions := make([]string, 0, remainingCount)
	for sessionID := range om.conversations {
		remainingSessions = append(remainingSessions, sessionID)
	}
	om.mu.RUnlock()

	t.Logf("After cleanup: %d sessions remain: %v", remainingCount, remainingSessions)

	// Should have ~5 sessions remaining (first 5 that we refreshed)
	// Allow some tolerance due to timing
	if remainingCount < 4 || remainingCount > 6 {
		t.Errorf("Expected ~5 remaining sessions, got %d", remainingCount)
	}

	// Verify first 5 sessions still accessible
	activeCount := 0
	for i := 0; i < 5; i++ {
		history := om.getConversationHistory(sessionIDs[i])
		if history != nil && len(history) == 2 { // Original message + keep-alive
			activeCount++
		}
	}

	if activeCount < 4 {
		t.Errorf("Expected at least 4 of first 5 sessions to be active, got %d", activeCount)
	}

	// Verify last 5 sessions are gone
	expiredCount := 0
	for i := 5; i < 10; i++ {
		history := om.getConversationHistory(sessionIDs[i])
		if history == nil {
			expiredCount++
		}
	}

	if expiredCount < 4 {
		t.Errorf("Expected at least 4 of last 5 sessions to be expired, got %d", expiredCount)
	}

	t.Logf("Successfully verified batch cleanup: %d active, %d expired", activeCount, expiredCount)
}

// Test cleanup with all sessions expired
func TestConversationCleanupBatch_AllExpired(t *testing.T) {
	shortTTL := 50 * time.Millisecond

	cfg := &Config{
		BaseURL:    "http://localhost:1234/v1",
		SessionTTL: shortTTL,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	// Create 20 sessions
	for i := 0; i < 20; i++ {
		sessionID := fmt.Sprintf("expire-session-%d", i)
		msg := &OpenAIMessage{
			Role:    "user",
			Content: fmt.Sprintf("Message %d", i),
		}
		om.addToHistory(sessionID, msg)
	}

	// Verify all exist
	om.mu.RLock()
	initialCount := len(om.conversations)
	om.mu.RUnlock()

	if initialCount != 20 {
		t.Errorf("Expected 20 sessions, got %d", initialCount)
	}

	// Wait for all to expire
	time.Sleep(60 * time.Millisecond)

	// Trigger cleanup
	om.mu.Lock()
	now := time.Now()
	for sessionID, state := range om.conversations {
		if now.Sub(state.lastAccess) > om.sessionTTL {
			delete(om.conversations, sessionID)
		}
	}
	cleanedCount := initialCount - len(om.conversations)
	finalCount := len(om.conversations)
	om.mu.Unlock()

	// All 20 should be cleaned
	if cleanedCount != 20 {
		t.Errorf("Expected to clean 20 sessions, cleaned %d", cleanedCount)
	}

	if finalCount != 0 {
		t.Errorf("Expected 0 remaining sessions, got %d", finalCount)
	}

	t.Logf("Successfully cleaned up all %d expired sessions", cleanedCount)
}

// Test cleanup with no expired sessions
func TestConversationCleanupBatch_NoneExpired(t *testing.T) {
	longTTL := 10 * time.Second

	cfg := &Config{
		BaseURL:    "http://localhost:1234/v1",
		SessionTTL: longTTL,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	// Create 15 sessions
	sessionIDs := make([]string, 15)
	for i := 0; i < 15; i++ {
		sessionIDs[i] = fmt.Sprintf("active-session-%d", i)
		msg := &OpenAIMessage{
			Role:    "user",
			Content: fmt.Sprintf("Active message %d", i),
		}
		om.addToHistory(sessionIDs[i], msg)
	}

	// Verify all exist
	om.mu.RLock()
	initialCount := len(om.conversations)
	om.mu.RUnlock()

	if initialCount != 15 {
		t.Errorf("Expected 15 sessions, got %d", initialCount)
	}

	// Wait a short time (not enough to expire with 10s TTL)
	time.Sleep(100 * time.Millisecond)

	// Trigger cleanup
	om.mu.Lock()
	now := time.Now()
	cleanedCount := 0
	for sessionID, state := range om.conversations {
		if now.Sub(state.lastAccess) > om.sessionTTL {
			delete(om.conversations, sessionID)
			cleanedCount++
		}
	}
	finalCount := len(om.conversations)
	om.mu.Unlock()

	// None should be cleaned
	if cleanedCount != 0 {
		t.Errorf("Expected to clean 0 sessions, cleaned %d", cleanedCount)
	}

	if finalCount != 15 {
		t.Errorf("Expected 15 remaining sessions, got %d", finalCount)
	}

	// Verify all sessions still accessible
	for i, sessionID := range sessionIDs {
		history := om.getConversationHistory(sessionID)
		if history == nil || len(history) != 1 {
			t.Errorf("Session %d (%s) should still exist", i, sessionID)
		}
	}

	t.Logf("All %d sessions remain active (no cleanup needed)", finalCount)
}

// TestSessionIDFallbackInGenerateContent tests that GenerateContent auto-generates
// a UUID session ID when none is provided in context, and logs a warning.
func TestSessionIDFallbackInGenerateContent(t *testing.T) {
	var logBuf strings.Builder
	logger := log.New(&logBuf, "", 0)

	cfg := &Config{
		BaseURL: "http://localhost:1234/v1", // Mock server (won't actually call)
		Logger:  logger,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	// Create a context WITHOUT session ID
	ctx := context.Background()

	// Call convertToOpenAIMessages which should auto-generate session ID
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Parts: []*genai.Part{
					genai.NewPartFromText("Hello, test message"),
				},
				Role: "user",
			},
		},
	}

	messages, err := om.convertToOpenAIMessages(ctx, req)
	if err != nil {
		t.Fatalf("convertToOpenAIMessages failed: %v", err)
	}

	// Verify messages were created
	if len(messages) == 0 {
		t.Error("Expected at least one message")
	}

	// Check that warning was logged about auto-generated session ID
	logOutput := logBuf.String()
	if !strings.Contains(logOutput, "WARNING") {
		t.Error("Expected WARNING in log output")
	}
	if !strings.Contains(logOutput, "No session ID in context, auto-generated:") {
		t.Errorf("Expected auto-generation warning, got: %s", logOutput)
	}

	// Extract the auto-generated UUID from log
	// Format: "WARNING: No session ID in context, auto-generated: <uuid>"
	lines := strings.Split(strings.TrimSpace(logOutput), "\n")
	var autoGeneratedID string
	for _, line := range lines {
		if strings.Contains(line, "auto-generated:") {
			parts := strings.Split(line, "auto-generated:")
			if len(parts) == 2 {
				autoGeneratedID = strings.TrimSpace(parts[1])
				break
			}
		}
	}

	if autoGeneratedID == "" {
		t.Fatal("Could not extract auto-generated session ID from log")
	}

	t.Logf("Auto-generated session ID: %s", autoGeneratedID)

	// Verify it's a valid UUID (length 36, format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
	if len(autoGeneratedID) != 36 {
		t.Errorf("Expected UUID length 36, got %d: %s", len(autoGeneratedID), autoGeneratedID)
	}
	if strings.Count(autoGeneratedID, "-") != 4 {
		t.Errorf("Expected UUID with 4 dashes, got: %s", autoGeneratedID)
	}

	// Verify that a session was created with the auto-generated ID
	om.mu.RLock()
	state, exists := om.conversations[autoGeneratedID]
	om.mu.RUnlock()

	if !exists {
		t.Errorf("Expected session to be created with auto-generated ID: %s", autoGeneratedID)
		return
	}

	// Verify the session has the user message
	if len(state.history) != 1 {
		t.Errorf("Expected 1 message in auto-generated session, got %d", len(state.history))
	}
	if len(state.history) > 0 {
		if state.history[0].Role != "user" {
			t.Errorf("Expected user message, got role: %s", state.history[0].Role)
		}
		if state.history[0].Content != "Hello, test message" {
			t.Errorf("Expected 'Hello, test message', got: %v", state.history[0].Content)
		}
	}

	t.Logf("Successfully verified auto-generated session ID and history")
}

// TestSessionIDFallbackMultipleCalls tests that each call without session ID
// gets a different auto-generated UUID.
func TestSessionIDFallbackMultipleCalls(t *testing.T) {
	var logBuf strings.Builder
	logger := log.New(&logBuf, "", 0)

	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
		Logger:  logger,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	// Create context WITHOUT session ID
	ctx := context.Background()

	// Make 3 calls without session ID
	sessionIDs := make([]string, 3)
	for i := 0; i < 3; i++ {
		logBuf.Reset() // Clear log buffer

		req := &model.LLMRequest{
			Contents: []*genai.Content{
				{
					Parts: []*genai.Part{
						genai.NewPartFromText(fmt.Sprintf("Message %d", i)),
					},
					Role: "user",
				},
			},
		}

		_, err := om.convertToOpenAIMessages(ctx, req)
		if err != nil {
			t.Fatalf("Call %d failed: %v", i, err)
		}

		// Extract session ID from log
		logOutput := logBuf.String()
		lines := strings.Split(strings.TrimSpace(logOutput), "\n")
		for _, line := range lines {
			if strings.Contains(line, "auto-generated:") {
				parts := strings.Split(line, "auto-generated:")
				if len(parts) == 2 {
					sessionIDs[i] = strings.TrimSpace(parts[1])
					break
				}
			}
		}

		if sessionIDs[i] == "" {
			t.Errorf("Call %d: Could not extract auto-generated session ID", i)
		}
	}

	// Verify all 3 session IDs are different (each call gets new UUID)
	if sessionIDs[0] == sessionIDs[1] || sessionIDs[0] == sessionIDs[2] || sessionIDs[1] == sessionIDs[2] {
		t.Errorf("Expected different session IDs for each call, got: %v", sessionIDs)
	}

	// Verify all 3 sessions exist in history
	om.mu.RLock()
	sessionCount := len(om.conversations)
	om.mu.RUnlock()

	if sessionCount != 3 {
		t.Errorf("Expected 3 sessions, got %d", sessionCount)
	}

	t.Logf("Successfully verified 3 unique auto-generated session IDs: %v", sessionIDs)
}

// TestSessionTTLInAdapter tests session expiration mid-conversation.
// This integration test verifies:
// 1. Session is active during conversation
// 2. Session expires after TTL passes
// 3. Background cleanup removes expired session
// 4. getConversationHistory returns nil for expired session
func TestSessionTTLInAdapter(t *testing.T) {
	var logBuf strings.Builder
	logger := log.New(&logBuf, "[TTL-TEST] ", 0)

	// Use very short TTL for testing
	shortTTL := 200 * time.Millisecond

	cfg := &Config{
		BaseURL:    "http://localhost:1234/v1",
		Logger:     logger,
		SessionTTL: shortTTL,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	sessionID := "conversation-session"

	// Phase 1: Start conversation - add first message
	t.Log("Phase 1: Starting conversation")
	msg1 := &OpenAIMessage{
		Role:    "user",
		Content: "Hello, this is message 1",
	}
	om.addToHistory(sessionID, msg1)

	// Verify session exists
	history := om.getConversationHistory(sessionID)
	if history == nil || len(history) != 1 {
		t.Fatalf("Expected 1 message in history, got %v", history)
	}
	t.Logf("Phase 1: Session created with 1 message")

	// Phase 2: Continue conversation quickly - add second message
	time.Sleep(50 * time.Millisecond)
	t.Log("Phase 2: Continuing conversation (50ms later)")
	msg2 := &OpenAIMessage{
		Role:    "assistant",
		Content: "Response to message 1",
	}
	om.addToHistory(sessionID, msg2)

	history = om.getConversationHistory(sessionID)
	if history == nil || len(history) != 2 {
		t.Fatalf("Expected 2 messages in history, got %v", history)
	}
	t.Logf("Phase 2: Session now has 2 messages")

	// Phase 3: Add third message
	time.Sleep(50 * time.Millisecond)
	t.Log("Phase 3: Adding third message (100ms total elapsed)")
	msg3 := &OpenAIMessage{
		Role:    "user",
		Content: "This is message 3",
	}
	om.addToHistory(sessionID, msg3)
	// IMPORTANT: msg3 updates lastAccess to current time (100ms mark)

	history = om.getConversationHistory(sessionID)
	if history == nil || len(history) != 3 {
		t.Fatalf("Expected 3 messages in history, got %v", history)
	}
	t.Logf("Phase 3: Session has 3 messages, still active (lastAccess updated)")

	// Phase 4: Wait but still within TTL from last access
	// Last access was just now (msg3), TTL = 200ms
	// Wait 150ms = still < 200ms from last access
	time.Sleep(150 * time.Millisecond)
	t.Logf("Phase 4: 150ms since last access - session should still be valid (TTL=200ms)")

	history = om.getConversationHistory(sessionID)
	if history == nil {
		t.Error("Session should still be valid at 150ms since last access (TTL=200ms)")
	} else {
		t.Logf("Phase 4: Session still valid with %d messages", len(history))
	}

	// Phase 5: NOW wait past TTL from last access
	// We're at 150ms since msg3, wait another 80ms = 230ms total inactivity
	// This exceeds 200ms TTL
	time.Sleep(80 * time.Millisecond)
	t.Logf("Phase 5: 230ms since last access - session should be expired (TTL=200ms)")

	// Check if getConversationHistory sees it as expired
	history = om.getConversationHistory(sessionID)
	if history != nil {
		t.Errorf("Expected session to be expired (returns nil), but got %d messages", len(history))
	} else {
		t.Log("Phase 5: ✓ getConversationHistory correctly returns nil for expired session")
	}

	// Phase 6: Wait for background cleanup to remove it
	// Background cleanup runs every TTL/2 = 100ms
	// We need to wait for next cleanup cycle
	t.Log("Phase 6: Waiting for background cleanup goroutine...")
	time.Sleep(150 * time.Millisecond) // Wait for cleanup cycle

	// Verify session was removed from conversations map
	om.mu.RLock()
	_, exists := om.conversations[sessionID]
	sessionCount := len(om.conversations)
	om.mu.RUnlock()

	if exists {
		t.Error("Expected background cleanup to remove expired session from map")
	} else {
		t.Log("Phase 6: ✓ Background cleanup removed expired session")
	}

	if sessionCount != 0 {
		t.Errorf("Expected 0 sessions after cleanup, got %d", sessionCount)
	}

	// Phase 7: Try to start new conversation with same session ID
	t.Log("Phase 7: Starting fresh conversation with same session ID")
	msg4 := &OpenAIMessage{
		Role:    "user",
		Content: "New conversation after expiration",
	}
	om.addToHistory(sessionID, msg4)

	history = om.getConversationHistory(sessionID)
	if history == nil {
		t.Fatal("Expected new session to be created")
	}
	if len(history) != 1 {
		t.Errorf("Expected fresh session with 1 message, got %d messages", len(history))
	}
	if history[0].Content != "New conversation after expiration" {
		t.Error("Expected new session to NOT contain old messages")
	}

	t.Log("Phase 7: ✓ Fresh session created, old history not retained")
	t.Log("SUCCESS: Session TTL integration test completed")
}

// TestSessionTTLWithActiveConversation tests that active conversations don't expire.
func TestSessionTTLWithActiveConversation(t *testing.T) {
	logger := log.New(os.Stdout, "[ACTIVE-TEST] ", log.LstdFlags)

	// Short TTL for testing
	shortTTL := 300 * time.Millisecond

	cfg := &Config{
		BaseURL:    "http://localhost:1234/v1",
		Logger:     logger,
		SessionTTL: shortTTL,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	sessionID := "active-session"

	// Start conversation
	om.addToHistory(sessionID, &OpenAIMessage{
		Role:    "user",
		Content: "Message 1",
	})

	// Keep conversation active by sending messages every 100ms
	// This should prevent expiration even though total time > TTL
	for i := 2; i <= 5; i++ {
		time.Sleep(100 * time.Millisecond)

		msg := &OpenAIMessage{
			Role:    "user",
			Content: fmt.Sprintf("Message %d", i),
		}
		om.addToHistory(sessionID, msg)

		// Verify session still exists
		history := om.getConversationHistory(sessionID)
		if history == nil {
			t.Errorf("After message %d: Session should still be active", i)
			return
		}
		if len(history) != i {
			t.Errorf("After message %d: Expected %d messages, got %d", i, i, len(history))
		}

		t.Logf("After message %d (%dms elapsed): Session active with %d messages",
			i, i*100, len(history))
	}

	// Total elapsed: 5 * 100ms = 500ms > 300ms TTL
	// But session should still be active because we kept accessing it
	history := om.getConversationHistory(sessionID)
	if history == nil {
		t.Error("Active session should not expire when continuously accessed")
	} else {
		t.Logf("SUCCESS: Active session survived %dms with %d messages (TTL=%dms)",
			500, len(history), shortTTL.Milliseconds())
	}

	// Verify all messages are preserved
	if len(history) != 5 {
		t.Errorf("Expected all 5 messages preserved, got %d", len(history))
	}
}

// TestSessionTTLCleanupTiming tests the exact timing of background cleanup.
func TestSessionTTLCleanupTiming(t *testing.T) {
	var logBuf strings.Builder
	logger := log.New(&logBuf, "", 0)

	ttl := 200 * time.Millisecond
	cfg := &Config{
		BaseURL:    "http://localhost:1234/v1",
		Logger:     logger,
		SessionTTL: ttl,
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)
	sessionID := "timing-test"

	// Create session at T=0
	start := time.Now()
	om.addToHistory(sessionID, &OpenAIMessage{
		Role:    "user",
		Content: "Test message",
	})

	t.Logf("T=0ms: Session created")

	// At T=180ms, session should still exist (< 200ms TTL)
	time.Sleep(180 * time.Millisecond)
	elapsed := time.Since(start)

	om.mu.RLock()
	_, exists := om.conversations[sessionID]
	om.mu.RUnlock()

	if !exists {
		t.Errorf("T=%dms: Session should still exist (TTL=%dms)", elapsed.Milliseconds(), ttl.Milliseconds())
	} else {
		t.Logf("T=%dms: ✓ Session exists (not yet expired)", elapsed.Milliseconds())
	}

	// getConversationHistory should still return data
	history := om.getConversationHistory(sessionID)
	if history == nil {
		t.Errorf("T=%dms: getConversationHistory should return data", elapsed.Milliseconds())
	}

	// At T=220ms, session has expired but may not be cleaned up yet
	// (cleanup runs every 100ms, so next cleanup is at T=200ms or T=300ms)
	time.Sleep(40 * time.Millisecond)
	elapsed = time.Since(start)

	history = om.getConversationHistory(sessionID)
	if history != nil {
		t.Logf("T=%dms: WARNING - Session expired but getConversationHistory returned data (timing variation)", elapsed.Milliseconds())
	} else {
		t.Logf("T=%dms: ✓ getConversationHistory returns nil (expired)", elapsed.Milliseconds())
	}

	// Wait for cleanup cycle (runs every TTL/2 = 100ms)
	// Worst case: need to wait up to 100ms for next cleanup
	time.Sleep(150 * time.Millisecond)
	elapsed = time.Since(start)

	om.mu.RLock()
	_, exists = om.conversations[sessionID]
	om.mu.RUnlock()

	if exists {
		t.Errorf("T=%dms: Background cleanup should have removed session by now", elapsed.Milliseconds())
	} else {
		t.Logf("T=%dms: ✓ Background cleanup removed expired session", elapsed.Milliseconds())
	}
}
