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
