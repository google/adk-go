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

package llminternal

import (
	"fmt"
	"strings"
	"time"

	"github.com/rs/zerolog/log"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// AudioCacheManager manages audio caching and flushing for live streaming flows.
type AudioCacheManager struct {
	config AudioCacheConfig
}

// AudioCacheConfig is the configuration for audio caching behavior.
type AudioCacheConfig struct {
	MaxCacheSizeBytes       int
	MaxCacheDurationSeconds float64
	AutoFlushThreshold      int // Number of chunks
}

// NewAudioCacheManager initializes the audio cache manager.
func NewAudioCacheManager(config *AudioCacheConfig) *AudioCacheManager {
	if config == nil {
		config = &AudioCacheConfig{
			MaxCacheSizeBytes:       10 * 1024 * 1024, // 10MB
			MaxCacheDurationSeconds: 300.0,            // 5 minutes
			AutoFlushThreshold:      100,
		}
	}
	return &AudioCacheManager{config: *config}
}

// CacheAudio caches incoming user or outgoing model audio data.
func (m *AudioCacheManager) CacheAudio(ctx agent.InvocationContext, audioBlob *genai.Blob, cacheType string) error {
	var role string
	var cacheSize int

	switch cacheType {
	case "input":
		role = "user"
		entry := agent.RealtimeCacheEntry{
			Role:      role,
			Data:      audioBlob,
			Timestamp: time.Now(),
		}
		ctx.AppendInputRealtimeCache(entry)
		cacheSize = len(ctx.InputRealtimeCache())
	case "output":
		role = "model"
		entry := agent.RealtimeCacheEntry{
			Role:      role,
			Data:      audioBlob,
			Timestamp: time.Now(),
		}
		ctx.AppendOutputRealtimeCache(entry)
		cacheSize = len(ctx.OutputRealtimeCache())
	default:
		return fmt.Errorf("cacheType must be either 'input' or 'output'")
	}

	log.Debug().
		Str("type", cacheType).
		Int("bytes", len(audioBlob.Data)).
		Int("cacheSize", cacheSize).
		Msg("Cached audio chunk")

	return nil
}

// FlushCaches flushes audio caches to artifact services.
// The multimodality data is saved in artifact service in the format of
// audio file. The file data reference is added to the session as an event.
// The audio file follows the naming convention: artifact_ref =
// f"artifact://{invocation_context.app_name}/{invocation_context.user_id}/
// {invocation_context.session.id}/_adk_live/{filename}#{revision_id}"

// Note: video data is not supported yet.

// Args:
// 	invocation_context: The invocation context containing audio caches.
// 	flush_user_audio: Whether to flush the input (user) audio cache.
// 	flush_model_audio: Whether to flush the output (model) audio cache.
// Returns:
// 	A list of Event objects created from the flushed caches.
func (m *AudioCacheManager) FlushCaches(ctx agent.InvocationContext, flushUserAudio, flushModelAudio bool) ([]*session.Event, error) {
	var flushedEvents []*session.Event

	if flushUserAudio && len(ctx.InputRealtimeCache()) > 0 {
		audioEvent, err := m.flushCacheToServices(ctx, ctx.InputRealtimeCache(), "input_audio")
		if err != nil {
			return nil, err
		}
		if audioEvent != nil {
			flushedEvents = append(flushedEvents, audioEvent)
			ctx.ClearInputRealtimeCache()
		}
	}

	if flushModelAudio && len(ctx.OutputRealtimeCache()) > 0 {
		log.Debug().Msg("Flushing output audio cache")
		audioEvent, err := m.flushCacheToServices(ctx, ctx.OutputRealtimeCache(), "output_audio")
		if err != nil {
			return nil, err
		}
		if audioEvent != nil {
			flushedEvents = append(flushedEvents, audioEvent)
			ctx.ClearOutputRealtimeCache()
		}
	}

	return flushedEvents, nil
}

func (m *AudioCacheManager) flushCacheToServices(ctx agent.InvocationContext, audioCache []agent.RealtimeCacheEntry, cacheType string) (*session.Event, error) {
	if ctx.Artifacts() == nil || len(audioCache) == 0 {
		log.Debug().Msg("Skipping cache flush: no artifact service or empty cache")
		return nil, nil
	}

	// Combine audio chunks into a single file
	var combinedAudioData []byte
	mimeType := "audio/pcm"
	if len(audioCache) > 0 && audioCache[0].Data != nil {
		mimeType = audioCache[0].Data.MIMEType
	}

	for _, entry := range audioCache {
		if entry.Data != nil {
			combinedAudioData = append(combinedAudioData, entry.Data.Data...)
		}
	}

	// Generate filename with timestamp from first audio chunk
	timestamp := audioCache[0].Timestamp.UnixMilli()
	ext := "pcm"
	parts := strings.Split(mimeType, "/")
	if len(parts) > 1 {
		ext = parts[len(parts)-1]
	}
	filename := fmt.Sprintf("adk_live_audio_storage_%s_%d.%s", cacheType, timestamp, ext)

	// Save to artifact service
	combinedAudioPart := &genai.Part{
		InlineData: &genai.Blob{
			Data:     combinedAudioData,
			MIMEType: mimeType,
		},
	}

	resp, err := ctx.Artifacts().Save(ctx, filename, combinedAudioPart)
	if err != nil {
		log.Error().Err(err).Str("type", cacheType).Msg("Failed to save artifact during flush")
		return nil, err
	}

	// Create artifact reference
	sess := ctx.Session()
	artifactRef := fmt.Sprintf("artifact://%s/%s/%s/_adk_live/%s#%d",
		sess.AppName(), sess.UserID(), sess.ID(), filename, resp.Version)

	// Create event
	author := audioCache[0].Role
	if audioCache[0].Role == "model" {
		author = ctx.Agent().Name()
	}

	audioEvent := session.NewEvent(ctx.InvocationID())
	audioEvent.Author = author
	audioEvent.Timestamp = audioCache[0].Timestamp
	audioEvent.Content = &genai.Content{
		Role: audioCache[0].Role,
		Parts: []*genai.Part{
			{
				FileData: &genai.FileData{
					FileURI:  artifactRef,
					MIMEType: mimeType,
				},
			},
		},
	}

	log.Debug().
		Str("type", cacheType).
		Int("chunks", len(audioCache)).
		Int("bytes", len(combinedAudioData)).
		Str("filename", filename).
		Msg("Successfully flushed audio cache")

	return audioEvent, nil
}

// GetCacheStats returns statistics about current cache state.
func (m *AudioCacheManager) GetCacheStats(ctx agent.InvocationContext) map[string]int {
	inputCache := ctx.InputRealtimeCache()
	outputCache := ctx.OutputRealtimeCache()

	inputCount := len(inputCache)
	outputCount := len(outputCache)

	inputBytes := 0
	for _, entry := range inputCache {
		if entry.Data != nil {
			inputBytes += len(entry.Data.Data)
		}
	}

	outputBytes := 0
	for _, entry := range outputCache {
		if entry.Data != nil {
			outputBytes += len(entry.Data.Data)
		}
	}

	return map[string]int{
		"input_chunks":  inputCount,
		"output_chunks": outputCount,
		"input_bytes":   inputBytes,
		"output_bytes":  outputBytes,
		"total_chunks":  inputCount + outputCount,
		"total_bytes":   inputBytes + outputBytes,
	}
}
