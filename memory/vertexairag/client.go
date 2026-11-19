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

package vertexairag

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"

	aiplatform "cloud.google.com/go/aiplatform/apiv1beta1"
	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"
	"google.golang.org/api/option/internaloption"
	htransport "google.golang.org/api/transport/http"

	aiplatformutil "google.golang.org/adk/v2/util/aiplatform"
)

// uploadFileName is the file name sent with each transcript. The upload
// endpoint uses its extension to detect the content type.
const uploadFileName = "transcript.txt"

type ragBackend interface {
	uploadFile(ctx context.Context, corpusName, displayName string, content io.Reader) error
	listFiles(ctx context.Context, req *aiplatformpb.ListRagFilesRequest) ([]*aiplatformpb.RagFile, string, error)
	retrieveContexts(ctx context.Context, req *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error)
}

// vertexRAGClient talks to Vertex AI over HTTP only. The protobuf
// UploadRagFileRequest has no field for the file bytes, so uploads require
// HTTP media upload; using the HTTP transport for every client lets the
// caller's options apply the same way to all of them.
type vertexRAGClient struct {
	ragClient      *aiplatform.VertexRagClient
	ragDataClient  *aiplatform.VertexRagDataClient
	httpClient     *http.Client
	uploadEndpoint string
}

func newVertexRAGClient(ctx context.Context, location string, opts ...option.ClientOption) (*vertexRAGClient, error) {
	ctx = context.WithoutCancel(ctx)
	for _, opt := range opts {
		if opt == nil {
			return nil, errors.New("client option cannot be nil")
		}
	}

	defaultEndpoint := "https://" + aiplatformutil.HostURL(location)
	// The regional endpoint goes first so that an option.WithEndpoint from the
	// caller replaces it.
	clientOpts := append([]option.ClientOption{
		option.WithEndpoint(defaultEndpoint),
	}, opts...)

	ragClient, err := aiplatform.NewVertexRagRESTClient(ctx, clientOpts...)
	if err != nil {
		return nil, fmt.Errorf("create VertexRagClient: %w", err)
	}
	ragDataClient, err := aiplatform.NewVertexRagDataRESTClient(ctx, clientOpts...)
	if err != nil {
		_ = ragClient.Close()
		return nil, fmt.Errorf("create VertexRagDataClient: %w", err)
	}
	httpOpts := append([]option.ClientOption{
		internaloption.WithDefaultEndpoint(defaultEndpoint),
		internaloption.WithDefaultScopes(aiplatform.DefaultAuthScopes()...),
	}, clientOpts...)
	httpClient, endpoint, err := htransport.NewClient(ctx, httpOpts...)
	if err != nil {
		_ = ragClient.Close()
		_ = ragDataClient.Close()
		return nil, fmt.Errorf("create upload HTTP client: %w", err)
	}

	return &vertexRAGClient{
		ragClient:      ragClient,
		ragDataClient:  ragDataClient,
		httpClient:     httpClient,
		uploadEndpoint: strings.TrimRight(endpoint, "/"),
	}, nil
}

type uploadMetadata struct {
	RagFile uploadRAGFile `json:"rag_file"`
}

type uploadRAGFile struct {
	DisplayName string `json:"display_name"`
}

type uploadResponse struct {
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// uploadFile sends a multipart/form-data request with a "metadata" part and a
// "file" part. The ragFiles:upload endpoint rejects the multipart/related body
// that the generated REST client sends.
func (c *vertexRAGClient) uploadFile(ctx context.Context, corpusName, displayName string, content io.Reader) error {
	metadata, err := json.Marshal(uploadMetadata{RagFile: uploadRAGFile{DisplayName: displayName}})
	if err != nil {
		return fmt.Errorf("marshal upload metadata: %w", err)
	}

	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	if err := mw.WriteField("metadata", string(metadata)); err != nil {
		return fmt.Errorf("write upload metadata: %w", err)
	}
	fw, err := mw.CreateFormFile("file", uploadFileName)
	if err != nil {
		return fmt.Errorf("create upload file part: %w", err)
	}
	if _, err := io.Copy(fw, content); err != nil {
		return fmt.Errorf("write upload file part: %w", err)
	}
	if err := mw.Close(); err != nil {
		return fmt.Errorf("close upload body: %w", err)
	}

	url := c.uploadEndpoint + "/upload/v1beta1/" + corpusName + "/ragFiles:upload"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, &body)
	if err != nil {
		return fmt.Errorf("create upload request: %w", err)
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	req.Header.Set("X-Goog-Upload-Protocol", "multipart")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if err := googleapi.CheckResponse(resp); err != nil {
		return err
	}
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read upload response: %w", err)
	}
	var parsed uploadResponse
	if len(bytes.TrimSpace(respBody)) > 0 {
		if err := json.Unmarshal(respBody, &parsed); err != nil {
			return fmt.Errorf("decode upload response: %w", err)
		}
	}
	if parsed.Error != nil {
		return fmt.Errorf("upload rag file failed with status code %d: %s", parsed.Error.Code, parsed.Error.Message)
	}
	return nil
}

// listFiles fetches one page of RAG files. InternalFetch is used instead of
// iterator.NewPager because NewPager loops across short or empty pages until
// PageSize items are buffered, and the generated REST fetch does not attach
// ctx to its HTTP request, which would let a server returning short pages
// exceed maxRAGFilePages requests.
func (c *vertexRAGClient) listFiles(ctx context.Context, req *aiplatformpb.ListRagFilesRequest) ([]*aiplatformpb.RagFile, string, error) {
	if err := ctx.Err(); err != nil {
		return nil, "", err
	}
	it := c.ragDataClient.ListRagFiles(ctx, req)
	return it.InternalFetch(int(req.GetPageSize()), req.GetPageToken())
}

func (c *vertexRAGClient) retrieveContexts(ctx context.Context, req *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error) {
	return c.ragClient.RetrieveContexts(ctx, req)
}
