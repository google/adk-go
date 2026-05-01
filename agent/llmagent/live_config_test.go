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

package llmagent

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
)

func TestLiveConfigFromRunConfig(t *testing.T) {
	t.Parallel()

	t.Run("nil_returns_nil", func(t *testing.T) {
		if got := liveConfigFromRunConfig(nil); got != nil {
			t.Errorf("liveConfigFromRunConfig(nil) = %v, want nil", got)
		}
	})

	t.Run("empty_config_returns_empty", func(t *testing.T) {
		got := liveConfigFromRunConfig(&agent.RunConfig{})
		want := &genai.LiveConnectConfig{}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("all_generation_params_mapped", func(t *testing.T) {
		temp := float32(0.7)
		topP := float32(0.9)
		topK := float32(40)
		maxTokens := int32(1024)

		rc := &agent.RunConfig{
			ThinkingConfig:  &genai.ThinkingConfig{IncludeThoughts: true},
			Temperature:     &temp,
			TopP:            &topP,
			TopK:            &topK,
			MaxOutputTokens: &maxTokens,
		}

		got := liveConfigFromRunConfig(rc)
		want := &genai.LiveConnectConfig{
			ThinkingConfig:  &genai.ThinkingConfig{IncludeThoughts: true},
			Temperature:     &temp,
			TopP:            &topP,
			TopK:            &topK,
			MaxOutputTokens: 1024,
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("nil_generation_params_not_mapped", func(t *testing.T) {
		rc := &agent.RunConfig{
			SpeechConfig: &genai.SpeechConfig{},
		}

		got := liveConfigFromRunConfig(rc)
		want := &genai.LiveConnectConfig{
			SpeechConfig: &genai.SpeechConfig{},
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("new_live_config_fields_mapped", func(t *testing.T) {
		enableAffective := true
		proactiveAudio := true
		targetTokens := int64(1024)
		rc := &agent.RunConfig{
			RealtimeInputConfig: &genai.RealtimeInputConfig{
				AutomaticActivityDetection: &genai.AutomaticActivityDetection{
					Disabled: true,
				},
			},
			Proactivity: &genai.ProactivityConfig{
				ProactiveAudio: &proactiveAudio,
			},
			EnableAffectiveDialog: &enableAffective,
			ContextWindowCompression: &genai.ContextWindowCompressionConfig{
				SlidingWindow: &genai.SlidingWindow{
					TargetTokens: &targetTokens,
				},
			},
			SessionResumption: &genai.SessionResumptionConfig{
				Handle: "prev-session-handle",
			},
		}

		got := liveConfigFromRunConfig(rc)
		want := &genai.LiveConnectConfig{
			RealtimeInputConfig:      rc.RealtimeInputConfig,
			Proactivity:              rc.Proactivity,
			EnableAffectiveDialog:    &enableAffective,
			ContextWindowCompression: rc.ContextWindowCompression,
			SessionResumption:        rc.SessionResumption,
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("new_live_config_fields_nil_not_mapped", func(t *testing.T) {
		rc := &agent.RunConfig{
			SpeechConfig: &genai.SpeechConfig{},
		}
		got := liveConfigFromRunConfig(rc)
		if got.RealtimeInputConfig != nil {
			t.Error("RealtimeInputConfig should be nil")
		}
		if got.Proactivity != nil {
			t.Error("Proactivity should be nil")
		}
		if got.EnableAffectiveDialog != nil {
			t.Error("EnableAffectiveDialog should be nil")
		}
		if got.ContextWindowCompression != nil {
			t.Error("ContextWindowCompression should be nil")
		}
		if got.SessionResumption != nil {
			t.Error("SessionResumption should be nil")
		}
	})

	t.Run("mixed_old_and_new_fields", func(t *testing.T) {
		temp := float32(0.5)
		enableAffective := false
		rc := &agent.RunConfig{
			// Existing fields
			Temperature:             &temp,
			SpeechConfig:            &genai.SpeechConfig{},
			InputAudioTranscription: true,
			// New fields — only two of five set
			EnableAffectiveDialog: &enableAffective,
			SessionResumption: &genai.SessionResumptionConfig{
				Handle: "resume-token-abc",
			},
		}

		got := liveConfigFromRunConfig(rc)
		want := &genai.LiveConnectConfig{
			Temperature:             &temp,
			SpeechConfig:            &genai.SpeechConfig{},
			InputAudioTranscription: &genai.AudioTranscriptionConfig{},
			EnableAffectiveDialog:   &enableAffective,
			SessionResumption:       &genai.SessionResumptionConfig{Handle: "resume-token-abc"},
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("mismatch (-want +got):\n%s", diff)
		}
		// Unset new fields must remain nil.
		if got.RealtimeInputConfig != nil {
			t.Error("RealtimeInputConfig should be nil when not set")
		}
		if got.Proactivity != nil {
			t.Error("Proactivity should be nil when not set")
		}
		if got.ContextWindowCompression != nil {
			t.Error("ContextWindowCompression should be nil when not set")
		}
	})
}
