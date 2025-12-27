package googlellm

import "testing"

func TestIsGemini2OrAbove(t *testing.T) {
	testCases := []struct {
		model string
		want  bool
	}{
		{"gemini-1.5-pro", false},
		{"gemini-2.0-flash", true},
		{"gemini-2.5-flash-lite", true},
		{"gemini-2.0-flash-exp", true},
		{"gemini-1.0-pro", false},
		{"projects/p/locations/l/models/gemini-2.0-flash", true},
		{"models/gemini-1.5-pro", false},
		{"not-a-gemini-model", false},
		{"gemini-2", true},
		{"gemini-3.0", true},
	}

	for _, tc := range testCases {
		got := IsGemini2OrAbove(tc.model)
		if got != tc.want {
			t.Errorf("IsGemini2OrAbove(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestIsGeminiModel(t *testing.T) {
	testCases := []struct {
		model string
		want  bool
	}{
		{"gemini-1.5-pro", true},
		{"models/gemini-2.0-flash", true},
		{"claud-3.5-sonnet", false},
	}

	for _, tc := range testCases {
		got := IsGeminiModel(tc.model)
		if got != tc.want {
			t.Errorf("IsGeminiModel(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestCanGeminiModelUseOutputSchemaWithTools(t *testing.T) {
	testCases := []struct {
		name   string
		model  string
		vertex bool
		want   bool
	}{
		{"Gemini1.5_Vertex", "gemini-1.5-pro", true, false},
		{"Gemini2.0_Vertex", "gemini-2.0-flash", true, true},
		{"Gemini2.0_GeminiAPI", "gemini-2.0-flash", false, false},
		{"NonGemini_Vertex", "not-a-gemini", true, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.vertex {
				t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")
			}
			got := CanGeminiModelUseOutputSchemaWithTools(tc.model)
			if got != tc.want {
				t.Errorf("CanGeminiModelUseOutputSchemaWithTools(%q) = %v, want %v", tc.model, got, tc.want)
			}
		})
	}
}
