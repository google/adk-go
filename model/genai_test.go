package model

import (
	"flag"
	"fmt"
	"strings"
	"testing"

	"github.com/google/adk-go"
	"google.golang.org/genai"
)

var manual = flag.Bool("manual", false, "Run manual tests that require a valid GenAI API key")

func TestNewGeminiModel(t *testing.T) {
	if !*manual {
		// TODO(hakim): remove this after making this test deterministic.
		t.Skip("Skipping manual test. Set -manual flag to run it.")
	}
	ctx := t.Context()
	modelName := "gemini-2.0-flash"
	m, err := NewGeminiModel(ctx, modelName, nil)
	if err != nil {
		t.Fatalf("NewGeminiModel(%q) failed: %v", modelName, err)
	}
	if got, want := m.Name(), modelName; got != want {
		t.Errorf("model Name = %q, want %q", got, want)
	}

	readResponse := func(s adk.LLMResponseStream) (string, error) {
		var answer string
		for resp, err := range s {
			if err != nil {
				return answer, err
			}
			if resp.Content == nil || len(resp.Content.Parts) == 0 {
				return answer, fmt.Errorf("encountered an empty response: %v", resp)
			}
			answer += resp.Content.Parts[0].Text
		}
		return answer, nil
	}

	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			s := m.GenerateContent(ctx, &adk.LLMRequest{
				Model:    m, // TODO: strange. What happens if this doesn't match m?
				Contents: genai.Text("What is the capital of France?"),
			}, stream)
			answer, err := readResponse(s)
			if err != nil || !strings.Contains(strings.ToLower(answer), "paris") {
				t.Errorf("GenerateContent(stream=%v)=(%q, %v), want ('.*paris.*', nil)", stream, answer, err)
			}
		})
	}
}
