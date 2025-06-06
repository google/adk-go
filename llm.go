package adk

import (
	"context"
	"iter"

	"google.golang.org/genai"
)

// Model is the LLM Model.
type Model interface {
	Name() string
	GenerateContent(ctx context.Context, req *LLMRequest, stream bool) (LLMResponseStream, error)
}

// LLMRequest is the input to LLMModel's generate functions.
// This allows passing in tools, output schema, and system instructions
// to the model.
type LLMRequest struct {
	Model          Model
	Contents       []*genai.Content
	GenerateConfig *genai.GenerateContentConfig

	// TODO: Can't we use genai's types?

	// Corresponds to adk-python's LLMRequest
	// TODO(jbd): Add other fields.
}

func (r *LLMRequest) AppendInstructions(instructions ...string) {
	panic("unimplemented")
}

func (r *LLMRequest) AppendTools(tools ...Tool) {
	panic("unimplemented")
}

// LLMResponseStream is the output of LLMModel's generate functions.
type LLMResponseStream iter.Seq2[*LLMResponse, error]

// LLMResponse provides the first candidate response from the model if available.
type LLMResponse struct {
	Content           *genai.Content
	GroundingMetadata *genai.GroundingMetadata
	// Partial indicates whether the content is part of a unfinished content stream.
	// Only used for streaming mode and when the content is plain text.
	Partial bool
	// Indicates whether the response from the model is complete.
	// Only used for streaming mode.
	TurnComplete bool
	// Flag indicating that LLM was interrupted when generating the content.
	// Usually it is due to user interruption during a bidi streaming.
	Interrupted bool
	// If
	ErrorCode    int
	ErrorMessage string
}
