package googlellm

import (
	"os"
	"slices"
)

const (
	// For using credentials from Google Vertex AI
	GoogleLLMVariantVertexAI = "VERTEX_AI"
	// For using API Key from Google AI Studio
	GoogleLLMVariantGeminiAPI = "GEMINI_API"
)

// GetGoogleLLMVariant returns the Google LLM variant to use.
// see https://google.github.io/adk-docs/get-started/quickstart/#set-up-the-model
func GetGoogleLLMVariant() string {
	useVertexAI, _ := os.LookupEnv("GOOGLE_GENAI_USE_VERTEXAI")
	if slices.Contains([]string{"1", "true"}, useVertexAI) {
		return GoogleLLMVariantVertexAI
	}
	return GoogleLLMVariantGeminiAPI
}
