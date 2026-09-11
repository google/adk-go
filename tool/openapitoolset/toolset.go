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

package openapitoolset

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/getkin/kin-openapi/openapi3"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/tool"
)

const (
	defaultToolsetName     = "openapi"
	defaultMaxResponseSize = int64(10 << 20)
	maxDocumentSize        = int64(10 << 20)
)

// Config controls how an OpenAPI document is exposed and executed.
type Config struct {
	// Name identifies the toolset. The default is "openapi".
	Name string
	// HTTPClient is used to load URL documents and execute operations. When nil,
	// http.DefaultClient is used. The client is copied before its transport is
	// wrapped, so Config is safe to reuse.
	HTTPClient *http.Client
	// Auth supplies a credential for each operation request. It is not used
	// while loading a document URL.
	Auth auth.CredentialProvider
	// ToolFilter controls which operations Tools exposes.
	ToolFilter tool.Predicate
	// BaseURL overrides the servers declared by the OpenAPI document.
	BaseURL string
	// MaxResponseBytes limits response bodies. The default is 10 MiB.
	MaxResponseBytes int64
}

// Toolset exposes the operations in an OpenAPI document as ADK tools.
type Toolset struct {
	name      string
	tools     []tool.Tool
	predicate tool.Predicate
}

// New creates a Toolset from an OpenAPI document encoded as JSON or YAML.
func New(ctx context.Context, document []byte, cfg Config) (*Toolset, error) {
	return newFromData(ctx, document, nil, cfg)
}

// NewFromFile creates a Toolset from an OpenAPI JSON or YAML file.
func NewFromFile(ctx context.Context, path string, cfg Config) (*Toolset, error) {
	if ctx == nil {
		return nil, fmt.Errorf("openapi toolset: context must not be nil")
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("openapi toolset: resolve document path: %w", err)
	}
	file, err := os.Open(absPath)
	if err != nil {
		return nil, fmt.Errorf("openapi toolset: read document: %w", err)
	}
	defer func() {
		_ = file.Close()
	}()
	document, tooLarge, err := readBounded(file, maxDocumentSize)
	if err != nil {
		return nil, fmt.Errorf("openapi toolset: read document: %w", err)
	}
	if tooLarge {
		return nil, fmt.Errorf("openapi toolset: document exceeds %d bytes", maxDocumentSize)
	}
	location := &url.URL{Scheme: "file", Path: filepath.ToSlash(absPath)}
	return newFromData(ctx, document, location, cfg)
}

// NewFromURL creates a Toolset from an OpenAPI JSON or YAML URL.
func NewFromURL(ctx context.Context, rawURL string, cfg Config) (*Toolset, error) {
	if ctx == nil {
		return nil, fmt.Errorf("openapi toolset: context must not be nil")
	}
	location, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("openapi toolset: parse URL: %w", err)
	}
	if location.Scheme != "http" && location.Scheme != "https" {
		return nil, fmt.Errorf("openapi toolset: URL scheme %q is not supported", location.Scheme)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, location.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("openapi toolset: create document request: %w", err)
	}
	client := configuredClient(cfg.HTTPClient)
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openapi toolset: load document: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("openapi toolset: load document: %s", resp.Status)
	}
	document, tooLarge, err := readBounded(resp.Body, maxDocumentSize)
	if err != nil {
		return nil, fmt.Errorf("openapi toolset: read document: %w", err)
	}
	if tooLarge {
		return nil, fmt.Errorf("openapi toolset: document exceeds %d bytes", maxDocumentSize)
	}
	return newFromData(ctx, document, location, cfg)
}

func newFromData(ctx context.Context, document []byte, location *url.URL, cfg Config) (*Toolset, error) {
	if ctx == nil {
		return nil, fmt.Errorf("openapi toolset: context must not be nil")
	}
	if len(document) == 0 {
		return nil, fmt.Errorf("openapi toolset: document is empty")
	}
	if int64(len(document)) > maxDocumentSize {
		return nil, fmt.Errorf("openapi toolset: document exceeds %d bytes", maxDocumentSize)
	}
	if cfg.MaxResponseBytes < 0 {
		return nil, fmt.Errorf("openapi toolset: MaxResponseBytes must not be negative")
	}

	loader := openapi3.NewLoader()
	loader.Context = ctx
	var (
		documentModel *openapi3.T
		err           error
	)
	if location == nil {
		documentModel, err = loader.LoadFromData(document)
	} else {
		documentModel, err = loader.LoadFromDataWithPath(document, location)
	}
	if err != nil {
		return nil, fmt.Errorf("openapi toolset: parse document: %w", err)
	}
	if err := documentModel.Validate(ctx); err != nil {
		return nil, fmt.Errorf("openapi toolset: validate document: %w", err)
	}

	maxResponseBytes := cfg.MaxResponseBytes
	if maxResponseBytes == 0 {
		maxResponseBytes = defaultMaxResponseSize
	}
	operations, err := buildOperationTools(documentModel, location, operationClient(cfg), cfg.BaseURL, maxResponseBytes)
	if err != nil {
		return nil, err
	}
	name := cfg.Name
	if name == "" {
		name = defaultToolsetName
	}
	return &Toolset{name: name, tools: operations, predicate: cfg.ToolFilter}, nil
}

func configuredClient(source *http.Client) *http.Client {
	if source == nil {
		source = http.DefaultClient
	}
	client := *source
	return &client
}

func operationClient(cfg Config) *http.Client {
	client := configuredClient(cfg.HTTPClient)
	if cfg.Auth != nil {
		base := client.Transport
		if base == nil {
			base = http.DefaultTransport
		}
		client.Transport = &auth.Transport{Provider: cfg.Auth, Base: base}
	}
	return client
}

// Name returns the configured toolset name.
func (t *Toolset) Name() string {
	return t.name
}

// Tools returns the generated operation tools in deterministic order.
func (t *Toolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	result := make([]tool.Tool, 0, len(t.tools))
	for _, operation := range t.tools {
		if t.predicate == nil || t.predicate(ctx, operation) {
			result = append(result, operation)
		}
	}
	return result, nil
}

// Close releases toolset resources. Toolset does not own the injected client,
// so Close is currently a no-op.
func (t *Toolset) Close() error {
	return nil
}

func buildOperationTools(document *openapi3.T, location *url.URL, client *http.Client, baseURLOverride string, maxResponseBytes int64) ([]tool.Tool, error) {
	if document.Paths == nil {
		return nil, fmt.Errorf("openapi toolset: document has no paths")
	}
	paths := document.Paths.Keys()
	sort.Strings(paths)
	seenNames := make(map[string]struct{})
	var result []tool.Tool
	for _, path := range paths {
		pathItem := document.Paths.Value(path)
		if pathItem == nil {
			continue
		}
		methods := make([]string, 0, len(pathItem.Operations()))
		for method := range pathItem.Operations() {
			methods = append(methods, method)
		}
		sort.Strings(methods)
		for _, method := range methods {
			operation := pathItem.Operations()[method]
			if operation == nil {
				continue
			}
			baseURL, err := operationBaseURL(baseURLOverride, operation.Servers, pathItem.Servers, document.Servers, location)
			if err != nil {
				return nil, fmt.Errorf("openapi toolset: %s %s: %w", strings.ToUpper(method), path, err)
			}
			generated, err := newOperationTool(strings.ToUpper(method), path, baseURL, pathItem.Parameters, operation, client, maxResponseBytes)
			if err != nil {
				return nil, fmt.Errorf("openapi toolset: %s %s: %w", strings.ToUpper(method), path, err)
			}
			if _, ok := seenNames[generated.Name()]; ok {
				return nil, fmt.Errorf("openapi toolset: duplicate operation tool name %q", generated.Name())
			}
			seenNames[generated.Name()] = struct{}{}
			result = append(result, generated)
		}
	}
	return result, nil
}

func operationBaseURL(override string, operationServers *openapi3.Servers, pathServers, documentServers openapi3.Servers, location *url.URL) (string, error) {
	if override != "" {
		return resolveServerURL(override, nil, location)
	}
	serverLists := []openapi3.Servers{documentServers}
	if len(pathServers) > 0 {
		serverLists = append(serverLists, pathServers)
	}
	if operationServers != nil && len(*operationServers) > 0 {
		serverLists = append(serverLists, *operationServers)
	}
	servers := serverLists[len(serverLists)-1]
	if len(servers) == 0 {
		return resolveServerURL("/", nil, location)
	}
	return resolveServerURL(servers[0].URL, servers[0].Variables, location)
}

func resolveServerURL(rawURL string, variables openapi3.ServerVariables, location *url.URL) (string, error) {
	for name, variable := range variables {
		if variable == nil {
			return "", fmt.Errorf("server variable %q is nil", name)
		}
		rawURL = strings.ReplaceAll(rawURL, "{"+name+"}", variable.Default)
	}
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return "", fmt.Errorf("parse server URL: %w", err)
	}
	if !parsed.IsAbs() {
		if location == nil || (location.Scheme != "http" && location.Scheme != "https") {
			return "", fmt.Errorf("relative server URL %q requires a URL document or BaseURL", rawURL)
		}
		parsed = location.ResolveReference(parsed)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", fmt.Errorf("server URL scheme %q is not supported", parsed.Scheme)
	}
	if parsed.Host == "" {
		return "", fmt.Errorf("server URL has no host")
	}
	if parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", fmt.Errorf("server URL must not contain a query or fragment")
	}
	return strings.TrimRight(parsed.String(), "/"), nil
}

func readBounded(reader io.Reader, limit int64) ([]byte, bool, error) {
	readLimit := limit
	if limit < int64(^uint64(0)>>1) {
		readLimit++
	}
	data, err := io.ReadAll(io.LimitReader(reader, readLimit))
	if err != nil {
		return nil, false, err
	}
	if int64(len(data)) > limit {
		return data[:limit], true, nil
	}
	return data, false, nil
}

var _ tool.Toolset = (*Toolset)(nil)
