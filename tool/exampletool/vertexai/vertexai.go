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

// Package vertexai provides an [exampletool.Provider] backed by a Vertex AI
// Example Store.
//
// A [Provider] holds one Example Store client for one store, named by its full
// resource name in [ProviderConfig]. Create it with [NewProvider], which
// connects to the store's regional endpoint with Application Default
// Credentials, or with [NewProviderWithOptions], which also applies the
// caller's client options (another endpoint, credentials, or an existing gRPC
// connection). Pass the Provider to the example tool:
//
//	p, err := vertexai.NewProvider(ctx, &vertexai.ProviderConfig{
//		ExampleStoreName: "projects/my-project/locations/us-central1/exampleStores/my-store",
//	})
//	if err != nil {
//		return err
//	}
//	defer p.(io.Closer).Close()
//	examples, err := exampletool.New(exampletool.ExampleToolConfig{Provider: p})
//
// On each model call the example tool calls [Provider.Examples] with the
// user's text, which runs one similarity search in the store and converts the
// results into [exampletool.Example] values. The number of results (10) and
// the similarity threshold (0.5) are fixed.
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

// The location is concatenated into the endpoint host, so its character set
// is narrow.
var exampleStoreNamePattern = regexp.MustCompile(`^projects/([a-zA-Z0-9._:-]+)/locations/([a-z0-9-]+)/exampleStores/([a-zA-Z0-9._-]+)$`)

// ProviderConfig configures a Vertex AI Example Store provider.
//
// The zero value is not valid; ExampleStoreName must be set to a full
// resource name. ProviderConfig is a plain value struct and is safe to copy
// and compare.
type ProviderConfig struct {
	// ExampleStoreName is the full resource name of the Vertex AI Example Store,
	// in the format "projects/{project}/locations/{location}/exampleStores/{example_store}".
	// The location picks the host the client dials, so it may contain only
	// lowercase letters, digits and '-'. Bare store names are rejected; the
	// provider does not read GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION from
	// the environment.
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

// Provider retrieves few-shot examples from a Vertex AI Example Store and
// implements [exampletool.Provider] and [io.Closer].
//
// Construct a Provider with [NewProvider] or [NewProviderWithOptions]; the
// zero value is not usable and its Examples method returns an error.
// A constructed Provider is safe for concurrent use by multiple goroutines,
// except that Close must not be called concurrently with Examples.
type Provider struct {
	storeName string
	project   string
	location  string
	client    exampleStoreClient
}

var (
	_ exampletool.Provider = (*Provider)(nil)
	_ io.Closer            = (*Provider)(nil)
)

// Close closes the underlying Vertex AI Example Store client. This also closes
// a connection passed to [NewProviderWithOptions] with [option.WithGRPCConn].
// After Close, Examples fails. Calling Close on a nil or zero Provider returns
// nil.
func (p *Provider) Close() error {
	if p == nil || p.client == nil {
		return nil
	}
	return p.client.Close()
}

// Project returns the Google Cloud project ID taken from the store's resource
// name, or "" if p is nil or uninitialized.
func (p *Provider) Project() string {
	if p == nil {
		return ""
	}
	return p.project
}

// Location returns the Google Cloud location taken from the store's resource
// name, or "" if p is nil or uninitialized. The client's default endpoint is
// the regional endpoint of this location.
func (p *Provider) Location() string {
	if p == nil {
		return ""
	}
	return p.location
}

// NewProvider creates an [exampletool.Provider] that retrieves few-shot examples
// from a Vertex AI Example Store.
//
// config must be non-nil and config.ExampleStoreName must be a full resource
// name of the form "projects/{project}/locations/{location}/exampleStores/{example_store}",
// with a location of lowercase letters, digits and '-'. Unlike adk-python,
// NewProvider does not fall back to GOOGLE_CLOUD_PROJECT or
// GOOGLE_CLOUD_LOCATION for bare store names.
//
// NewProvider creates the underlying Vertex AI Example Store gRPC client
// using Application Default Credentials and the regional endpoint derived from
// the store's location. Client creation is detached from ctx's cancellation via
// [context.WithoutCancel]. If config is invalid or client creation fails (for
// example, when default credentials cannot be found), NewProvider returns nil
// and a non-nil error.
//
// The returned provider is a [*Provider], which implements [io.Closer]. Close
// it when it is no longer in use to release the underlying client connection.
func NewProvider(ctx context.Context, config *ProviderConfig) (exampletool.Provider, error) {
	// Returning p directly on error would wrap a nil *Provider in a non-nil
	// interface value.
	p, err := NewProviderWithOptions(ctx, config)
	if err != nil {
		return nil, err
	}
	return p, nil
}

// NewProviderWithOptions creates a [*Provider] that retrieves few-shot examples
// from a Vertex AI Example Store, applying opts to the underlying Vertex AI
// Example Store client.
//
// config must be non-nil and config.ExampleStoreName must be a full resource
// name of the form "projects/{project}/locations/{location}/exampleStores/{example_store}",
// with a location of lowercase letters, digits and '-'. Unlike adk-python,
// NewProviderWithOptions does not fall back to GOOGLE_CLOUD_PROJECT or
// GOOGLE_CLOUD_LOCATION for bare store names.
//
// The regional endpoint derived from the store's location
// ("<location>-aiplatform.googleapis.com:443") is passed before opts, so a
// caller's [option.WithEndpoint] or [option.WithGRPCConn] overrides it, and
// credentials come from Application Default Credentials unless opts specify
// otherwise (for example, [option.WithAuthCredentials] or
// [option.WithoutAuthentication]). Client creation is detached from ctx's
// cancellation via [context.WithoutCancel].
//
// If config is nil, config.ExampleStoreName is invalid, any element of opts is
// nil, or client creation fails, NewProviderWithOptions returns nil and a
// non-nil error.
//
// Close the returned [*Provider] when it is no longer in use to release the
// underlying client connection. The client takes ownership of a connection
// passed with [option.WithGRPCConn]: Close closes it.
func NewProviderWithOptions(ctx context.Context, config *ProviderConfig, opts ...option.ClientOption) (*Provider, error) {
	project, location, err := resolveStoreResource(config)
	if err != nil {
		return nil, err
	}
	for _, opt := range opts {
		if opt == nil {
			return nil, errors.New("vertexai: client option cannot be nil")
		}
	}
	allOpts := append([]option.ClientOption{option.WithEndpoint(aiplatformutil.HostPortURL(location))}, opts...)
	client, err := newExampleStoreClient(context.WithoutCancel(ctx), allOpts...)
	if err != nil {
		return nil, fmt.Errorf("vertexai: create example store client: %w", err)
	}
	return &Provider{storeName: config.ExampleStoreName, project: project, location: location, client: client}, nil
}

func newProvider(config *ProviderConfig, client exampleStoreClient) (exampletool.Provider, error) {
	project, location, err := resolveStoreResource(config)
	if err != nil {
		return nil, err
	}
	if client == nil {
		return nil, errors.New("vertexai: client cannot be nil")
	}
	return &Provider{
		storeName: config.ExampleStoreName,
		project:   project,
		location:  location,
		client:    client,
	}, nil
}

func parseLocation(config *ProviderConfig) (string, error) {
	_, location, err := resolveStoreResource(config)
	return location, err
}

// Unlike adk-python's VertexAiExampleStore, which falls back to
// GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION for bare store IDs,
// resolveStoreResource requires a full resource name.
func resolveStoreResource(config *ProviderConfig) (project, location string, err error) {
	if config == nil {
		return "", "", errors.New("vertexai: config cannot be nil")
	}
	matches := exampleStoreNamePattern.FindStringSubmatch(config.ExampleStoreName)
	if len(matches) != 4 {
		return "", "", errors.New("vertexai: ExampleStoreName must match projects/{project}/locations/{location}/exampleStores/{example_store}")
	}
	return matches[1], matches[2], nil
}

// Examples retrieves few-shot examples matching query from the configured
// Vertex AI Example Store, implementing [exampletool.Provider].
//
// Examples sends one SearchExamples RPC for the configured store with top_k=10
// and query as the last-entry user content search key. Results with
// similarity_score < 0.5 or without a stored_contents_example are skipped;
// the others are returned in the order of the response. Each returned
// Example's Input is a user content with one text part holding the stored
// example's search key, and its Output holds the stored expected contents in
// order, converted to [genai.Content]. Unset contents are left out, and so are
// parts that carry no data and no thought flag, thought signature or video
// metadata.
//
// If the search returns no matching results, Examples returns an empty,
// non-nil slice and a nil error. If the RPC fails or p is uninitialized,
// Examples returns nil and a non-nil error wrapping the underlying cause.
func (p *Provider) Examples(ctx context.Context, query string) ([]*exampletool.Example, error) {
	if p == nil || p.client == nil {
		return nil, errors.New("vertexai: provider is not initialized")
	}
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
					ID:          fc.GetId(),
					Name:        fc.GetName(),
					Args:        fc.GetArgs().AsMap(),
					PartialArgs: convertPartialArgs(fc.GetPartialArgs()),
				}
				// genai.FunctionCall.WillContinue is a *bool; unset stays nil.
				if fc.GetWillContinue() {
					part.FunctionCall.WillContinue = new(true)
				}
			}
		case *aiplatformpb.Part_FunctionResponse:
			if fr := data.FunctionResponse; fr != nil {
				part.FunctionResponse = &genai.FunctionResponse{
					ID:       fr.GetId(),
					Name:     fr.GetName(),
					Response: fr.GetResponse().AsMap(),
					Parts:    convertFunctionResponseParts(fr.GetParts()),
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

func convertPartialArgs(src []*aiplatformpb.PartialArg) []*genai.PartialArg {
	if len(src) == 0 {
		return nil
	}
	args := make([]*genai.PartialArg, 0, len(src))
	for _, srcArg := range src {
		if srcArg == nil {
			continue
		}
		dstArg := &genai.PartialArg{JsonPath: srcArg.GetJsonPath()}
		if srcArg.GetWillContinue() {
			dstArg.WillContinue = new(true)
		}
		switch d := srcArg.GetDelta().(type) {
		case *aiplatformpb.PartialArg_NullValue:
			dstArg.NULLValue = d.NullValue.String()
		case *aiplatformpb.PartialArg_NumberValue:
			dstArg.NumberValue = new(d.NumberValue)
		case *aiplatformpb.PartialArg_StringValue:
			dstArg.StringValue = d.StringValue
		case *aiplatformpb.PartialArg_BoolValue:
			dstArg.BoolValue = new(d.BoolValue)
		}
		args = append(args, dstArg)
	}
	return args
}

func convertFunctionResponseParts(src []*aiplatformpb.FunctionResponsePart) []*genai.FunctionResponsePart {
	if len(src) == 0 {
		return nil
	}
	parts := make([]*genai.FunctionResponsePart, 0, len(src))
	for _, srcPart := range src {
		switch {
		case srcPart.GetInlineData() != nil:
			blob := srcPart.GetInlineData()
			parts = append(parts, &genai.FunctionResponsePart{InlineData: &genai.FunctionResponseBlob{
				MIMEType:    blob.GetMimeType(),
				Data:        blob.GetData(),
				DisplayName: blob.GetDisplayName(),
			}})
		case srcPart.GetFileData() != nil:
			fd := srcPart.GetFileData()
			parts = append(parts, &genai.FunctionResponsePart{FileData: &genai.FunctionResponseFileData{
				MIMEType:    fd.GetMimeType(),
				FileURI:     fd.GetFileUri(),
				DisplayName: fd.GetDisplayName(),
			}})
		}
	}
	return parts
}
