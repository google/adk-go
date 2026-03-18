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

		if got.ThinkingConfig != nil {
			t.Error("ThinkingConfig should be nil")
		}
		if got.Temperature != nil {
			t.Error("Temperature should be nil")
		}
		if got.TopP != nil {
			t.Error("TopP should be nil")
		}
		if got.TopK != nil {
			t.Error("TopK should be nil")
		}
		if got.MaxOutputTokens != 0 {
			t.Errorf("MaxOutputTokens = %d, want 0", got.MaxOutputTokens)
		}
		// SpeechConfig should still be mapped
		if got.SpeechConfig == nil {
			t.Error("SpeechConfig should be set")
		}
	})
}
