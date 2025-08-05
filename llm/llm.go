package llm

import (
	"context"
	"iter"

	"google.golang.org/genai"
)

type Model interface {
	Name() string
	Generate(ctx context.Context, req *Request) (*Response, error)
	GenerateStream(ctx context.Context, req *Request) iter.Seq2[*Response, error]
}

type Request struct {
	Contents       []*genai.Content
	GenerateConfig *genai.GenerateContentConfig
}

type Response struct {
	Content           *genai.Content
	GroundingMetadata *genai.GroundingMetadata
	Partial           bool
	TurnComplete      bool
	Interrupted       bool
	ErrorCode         int
	ErrorMessage      string
}
