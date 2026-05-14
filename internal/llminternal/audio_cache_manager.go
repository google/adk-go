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
	"fmt"
	"sync"
	"time"

	"google.golang.org/genai"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// AudioCacheManager manages audio caching and flushing for live streaming flows.
type AudioCacheManager struct {
	mu              sync.Mutex
	inputCache      [][]byte
	outputCache     [][]byte
	inputStartTime  time.Time
	outputStartTime time.Time
	mimeType        string
}

// NewAudioCacheManager creates a new AudioCacheManager.
func NewAudioCacheManager() *AudioCacheManager {
	return &AudioCacheManager{
		mimeType: "audio/pcm", // Default to audio/pcm
	}
}

// CacheInput caches incoming user audio data.
func (m *AudioCacheManager) CacheInput(data []byte, mimeType string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.inputCache) == 0 {
		m.inputStartTime = time.Now()
		if mimeType != "" {
			m.mimeType = mimeType
		}
	}
	m.inputCache = append(m.inputCache, data)
}

// CacheOutput caches outgoing model audio data.
func (m *AudioCacheManager) CacheOutput(data []byte, mimeType string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.outputCache) == 0 {
		m.outputStartTime = time.Now()
		if mimeType != "" {
			m.mimeType = mimeType
		}
	}
	m.outputCache = append(m.outputCache, data)
}

// FlushCaches flushes audio caches to artifact services and returns created events.
func (m *AudioCacheManager) FlushCaches(ctx agent.InvocationContext, flushUser, flushModel bool) ([]*session.Event, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	var events []*session.Event

	if flushUser && len(m.inputCache) > 0 {
		ev, err := m.flushCache(ctx, m.inputCache, "input_audio", m.inputStartTime)
		if err != nil {
			return nil, err
		}
		if ev != nil {
			events = append(events, ev)
		}
		m.inputCache = nil
	}

	if flushModel && len(m.outputCache) > 0 {
		ev, err := m.flushCache(ctx, m.outputCache, "output_audio", m.outputStartTime)
		if err != nil {
			return nil, err
		}
		if ev != nil {
			events = append(events, ev)
		}
		m.outputCache = nil
	}

	return events, nil
}

func (m *AudioCacheManager) flushCache(ctx agent.InvocationContext, cache [][]byte, cacheType string, startTime time.Time) (*session.Event, error) {
	if len(cache) == 0 {
		return nil, nil
	}

	// Combine chunks
	var buf bytes.Buffer
	for _, chunk := range cache {
		buf.Write(chunk)
	}
	combinedData := buf.Bytes()

	// Generate filename
	timestamp := startTime.UnixMilli()
	ext := "pcm"
	if m.mimeType == "audio/mp3" {
		ext = "mp3"
	}

	filename := fmt.Sprintf("adk_live_audio_storage_%s_%d.%s", cacheType, timestamp, ext)

	// Save to artifact service
	part := genai.NewPartFromBytes(combinedData, m.mimeType)
	resp, err := ctx.Artifacts().Save(ctx, filename, part)
	if err != nil {
		return nil, fmt.Errorf("failed to save audio artifact: %w", err)
	}

	// Create artifact reference
	sess := ctx.Session()
	artifactRef := fmt.Sprintf("artifact://%s/%s/%s/_adk_live/%s#%d", sess.AppName(), sess.UserID(), sess.ID(), filename, resp.Version)

	// Create event with file data reference
	author := ctx.Agent().Name()
	role := "model"
	if cacheType == "input_audio" {
		author = "user"
		role = "user"
	}

	ev := session.NewEvent(ctx.InvocationID())
	ev.Author = author
	ev.Timestamp = startTime
	ev.Content = &genai.Content{
		Role: role,
		Parts: []*genai.Part{
			{
				FileData: &genai.FileData{
					FileURI:  artifactRef,
					MIMEType: m.mimeType,
				},
			},
		},
	}

	return ev, nil
}
