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
}
