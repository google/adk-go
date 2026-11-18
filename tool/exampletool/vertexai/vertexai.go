// Copyright 2026 Google LLC
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

// Package vertexai provides an [exampletool.Provider] backed by a Vertex AI Example Store.
package vertexai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"regexp"

	aiplatform "cloud.google.com/go/aiplatform/apiv1beta1"
	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"google.golang.org/api/option"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/tool/exampletool"
	aiplatformutil "google.golang.org/adk/v2/util/aiplatform"
)

const (
	topK = 10
	// Below this threshold an example is more likely to mislead the model than help it.
	similarityThreshold = 0.5
)

var exampleStoreNamePattern = regexp.MustCompile(`^projects/([a-zA-Z0-9._:-]+)/locations/([a-zA-Z0-9._-]+)/exampleStores/([a-zA-Z0-9._-]+)$`)

// ProviderConfig configures a Vertex AI Example Store provider.
type ProviderConfig struct {
	// ExampleStoreName is the full resource name of the Vertex AI Example Store,
	// in the format "projects/{project}/locations/{location}/exampleStores/{example_store}".
	ExampleStoreName string
}

type exampleStoreClient interface {
	SearchExamples(ctx context.Context, req *aiplatformpb.SearchExamplesRequest) (*aiplatformpb.SearchExamplesResponse, error)
	Close() error
}

type clientAdapter struct {
	*aiplatform.ExampleStoreClient
}

func (a clientAdapter) SearchExamples(ctx context.Context, req *aiplatformpb.SearchExamplesRequest) (*aiplatformpb.SearchExamplesResponse, error) {
	return a.ExampleStoreClient.SearchExamples(ctx, req)
}

var newExampleStoreClient = func(ctx context.Context, opts ...option.ClientOption) (exampleStoreClient, error) {
	c, err := aiplatform.NewExampleStoreClient(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return clientAdapter{c}, nil
}

type provider struct {
	storeName string
	client    exampleStoreClient
}

var _ io.Closer = (*provider)(nil)

// Close closes the underlying Vertex AI Example Store client.
func (p *provider) Close() error {
	return p.client.Close()
}

// NewProvider creates an [exampletool.Provider] that retrieves few-shot examples
// from a Vertex AI Example Store.
//
// The returned provider implements [io.Closer]. Close it when it is no longer
// in use to release the underlying client connection.
func NewProvider(ctx context.Context, config *ProviderConfig) (exampletool.Provider, error) {
	ctx = context.WithoutCancel(ctx)
	location, err := parseLocation(config)
	if err != nil {
		return nil, err
	}
	client, err := newExampleStoreClient(ctx, option.WithEndpoint(aiplatformutil.HostPortURL(location)))
	if err != nil {
		return nil, fmt.Errorf("vertexai: create example store client: %w", err)
	}
	return newProvider(config, client)
}

func newProvider(config *ProviderConfig, client exampleStoreClient) (exampletool.Provider, error) {
	if _, err := parseLocation(config); err != nil {
		return nil, err
	}
	if client == nil {
		return nil, errors.New("vertexai: client cannot be nil")
	}
	return &provider{
		storeName: config.ExampleStoreName,
		client:    client,
	}, nil
}

// Unlike adk-python's VertexAiExampleStore, which falls back to
// GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION for bare store IDs,
// parseLocation requires a full resource name.
func parseLocation(config *ProviderConfig) (string, error) {
	if config == nil {
		return "", errors.New("vertexai: config cannot be nil")
	}
	matches := exampleStoreNamePattern.FindStringSubmatch(config.ExampleStoreName)
	if len(matches) != 4 {
		return "", errors.New("vertexai: ExampleStoreName must match projects/{project}/locations/{location}/exampleStores/{example_store}")
	}
	return matches[2], nil
}

// Examples implements [exampletool.Provider].
func (p *provider) Examples(ctx context.Context, query string) ([]*exampletool.Example, error) {
	req := &aiplatformpb.SearchExamplesRequest{
		ExampleStore: p.storeName,
		TopK:         topK,
		Parameters: &aiplatformpb.SearchExamplesRequest_StoredContentsExampleParameters{
			StoredContentsExampleParameters: &aiplatformpb.StoredContentsExampleParameters{
				Query: &aiplatformpb.StoredContentsExampleParameters_ContentSearchKey_{
					ContentSearchKey: &aiplatformpb.StoredContentsExampleParameters_ContentSearchKey{
						Contents: []*aiplatformpb.Content{
							{
								Role: genai.RoleUser,
								Parts: []*aiplatformpb.Part{
									{
										Data: &aiplatformpb.Part_Text{Text: query},
									},
								},
							},
						},
						SearchKeyGenerationMethod: &aiplatformpb.StoredContentsExample_SearchKeyGenerationMethod{
							Method: &aiplatformpb.StoredContentsExample_SearchKeyGenerationMethod_LastEntry_{
								LastEntry: &aiplatformpb.StoredContentsExample_SearchKeyGenerationMethod_LastEntry{},
							},
						},
					},
				},
			},
		},
	}

	resp, err := p.client.SearchExamples(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("vertexai: search examples: %w", err)
	}

	examples := make([]*exampletool.Example, 0)
	for _, result := range resp.GetResults() {
		if result.GetSimilarityScore() < similarityThreshold {
			continue
		}
		stored := result.GetExample().GetStoredContentsExample()
		if stored == nil {
			continue
		}
		expectedContents := stored.GetContentsExample().GetExpectedContents()
		output := make([]*genai.Content, 0, len(expectedContents))
		for _, expected := range expectedContents {
			if c := convertContent(expected.GetContent()); c != nil {
				output = append(output, c)
			}
		}
		examples = append(examples, &exampletool.Example{
			Input:  genai.NewContentFromText(stored.GetSearchKey(), genai.RoleUser),
			Output: output,
		})
	}
	return examples, nil
}

func convertContent(src *aiplatformpb.Content) *genai.Content {
	if src == nil {
		return nil
	}
	parts := make([]*genai.Part, 0, len(src.GetParts()))
	for _, srcPart := range src.GetParts() {
		if srcPart == nil {
			continue
		}
		part := &genai.Part{
			Thought:          srcPart.GetThought(),
			ThoughtSignature: srcPart.GetThoughtSignature(),
		}
		if vm := srcPart.GetVideoMetadata(); vm != nil {
			part.VideoMetadata = &genai.VideoMetadata{
				StartOffset: vm.GetStartOffset().AsDuration(),
				EndOffset:   vm.GetEndOffset().AsDuration(),
			}
		}
		switch data := srcPart.GetData().(type) {
		case *aiplatformpb.Part_Text:
			part.Text = data.Text
		case *aiplatformpb.Part_FunctionCall:
			if fc := data.FunctionCall; fc != nil {
				part.FunctionCall = &genai.FunctionCall{
					ID:   fc.GetId(),
					Name: fc.GetName(),
					Args: fc.GetArgs().AsMap(),
				}
			}
		case *aiplatformpb.Part_FunctionResponse:
			if fr := data.FunctionResponse; fr != nil {
				part.FunctionResponse = &genai.FunctionResponse{
					ID:       fr.GetId(),
					Name:     fr.GetName(),
					Response: fr.GetResponse().AsMap(),
				}
			}
		case *aiplatformpb.Part_InlineData:
			if blob := data.InlineData; blob != nil {
				part.InlineData = &genai.Blob{
					MIMEType: blob.GetMimeType(),
					Data:     blob.GetData(),
				}
			}
		case *aiplatformpb.Part_FileData:
			if fd := data.FileData; fd != nil {
				part.FileData = &genai.FileData{
					MIMEType: fd.GetMimeType(),
					FileURI:  fd.GetFileUri(),
				}
			}
		case *aiplatformpb.Part_ExecutableCode:
			if ec := data.ExecutableCode; ec != nil {
				part.ExecutableCode = &genai.ExecutableCode{
					Code:     ec.GetCode(),
					Language: genai.Language(ec.GetLanguage().String()),
				}
			}
		case *aiplatformpb.Part_CodeExecutionResult:
			if cer := data.CodeExecutionResult; cer != nil {
				part.CodeExecutionResult = &genai.CodeExecutionResult{
					Outcome: genai.Outcome(cer.GetOutcome().String()),
					Output:  cer.GetOutput(),
				}
			}
		default:
			if !part.Thought && len(part.ThoughtSignature) == 0 && part.VideoMetadata == nil {
				continue
			}
		}
		parts = append(parts, part)
	}
	return &genai.Content{
		Role:  src.GetRole(),
		Parts: parts,
	}
}
