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

// Package apigee provides an LLM implementation for calling Apigee proxy.
package apigee

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"os"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
)

const (
	apigeeProxyURLEnvVar         = "APIGEE_PROXY_URL"
	googleGenaiUseVertexAIEnvVar = "GOOGLE_GENAI_USE_VERTEXAI"
	projectEnvVar                = "GOOGLE_CLOUD_PROJECT"
	locationEnvVar               = "GOOGLE_CLOUD_LOCATION"
)

type modelInfo struct {
	modelID    string
	apiVersion string
	isVertexAI bool
}

type apigeeModel struct {
	delegate model.LLM
	name     string
}

// Config contains the configuration for the Apigee LLM.
type Config struct {
	ModelName     string
	ProxyURL      string
	CustomHeaders http.Header
	HTTPClient    *http.Client // For testing only.
}

// Option is a function that configures the Apigee LLM.
type Option func(*Config)

// WithProxyURL sets the proxy URL for the Apigee LLM.
func WithProxyURL(proxyURL string) Option {
	return func(c *Config) {
		c.ProxyURL = proxyURL
	}
}

// WithCustomHeaders sets the custom headers for the Apigee LLM.
func WithCustomHeaders(headers http.Header) Option {
	return func(c *Config) {
		c.CustomHeaders = headers
	}
}

// WithHTTPClient sets the HTTP client for the Apigee LLM. This is for testing only.
func WithHTTPClient(client *http.Client) Option {
	return func(c *Config) {
		c.HTTPClient = client
	}
}

// NewModel creates and initializes a new model instance that satisfies the
// model.LLM interface, backed by the Apigee proxy.
func NewModel(ctx context.Context, modelName string, opts ...Option) (*apigeeModel, error) {
	cfg := &Config{
		ModelName: modelName,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	if !strings.HasPrefix(cfg.ModelName, "apigee/") {
		return nil, fmt.Errorf("invalid model string: %s", cfg.ModelName)
	}
	mi, err := parseModelName(cfg.ModelName)
	if err != nil {
		return nil, err
	}

	proxyURL := resolveProxyURL(cfg.ProxyURL)
	if proxyURL == "" {
		return nil, fmt.Errorf("%s environment variable not set", apigeeProxyURLEnvVar)
	}

	httpOptions := generateHTTPOptions(proxyURL, mi.apiVersion, cfg.CustomHeaders)

	backendType := backendType(mi.isVertexAI)

	clientConfig, err := generateClientConfig(mi.isVertexAI, backendType, httpOptions, cfg.HTTPClient)
	if err != nil {
		return nil, err
	}

	delegate, err := gemini.NewModel(ctx, mi.modelID, clientConfig)
	if err != nil {
		return nil, err
	}

	return &apigeeModel{
		delegate: delegate,
		name:     cfg.ModelName,
	}, nil
}

func (m *apigeeModel) Name() string {
	return m.name
}

// GenerateContent calls the underlying model.
func (m *apigeeModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return m.delegate.GenerateContent(ctx, req, stream)
}

func parseModelName(modelName string) (*modelInfo, error) {
	if !strings.HasPrefix(modelName, "apigee/") {
		return nil, fmt.Errorf("invalid model string: %s", modelName)
	}
	modelPart := strings.TrimPrefix(modelName, "apigee/")
	if modelPart == "" {
		return nil, fmt.Errorf("invalid model string: %s", modelName)
	}
	components := strings.Split(modelPart, "/")

	info := &modelInfo{}
	info.isVertexAI = !strings.HasPrefix(modelName, "apigee/gemini/") &&
		(strings.HasPrefix(modelName, "apigee/vertex_ai/") ||
			isEnabled(googleGenaiUseVertexAIEnvVar))

	validated := false
	if len(components) == 1 {
		info.modelID = components[0]
		validated = true
	} else if len(components) == 2 {
		if components[0] == "vertex_ai" || components[0] == "gemini" {
			info.modelID = components[1]
			validated = true
		} else if strings.HasPrefix(components[0], "v") {
			info.apiVersion = components[0]
			info.modelID = components[1]
			validated = true
		}
	} else if len(components) == 3 {
		if (components[0] == "vertex_ai" || components[0] == "gemini") && strings.HasPrefix(components[1], "v") {
			info.apiVersion = components[1]
			info.modelID = components[2]
			validated = true
		}
	}

	if !validated {
		return nil, fmt.Errorf("invalid model string: %s", modelName)
	}

	return info, nil
}

func resolveProxyURL(proxyURL string) string {
	if proxyURL != "" {
		return proxyURL
	}
	return os.Getenv(apigeeProxyURLEnvVar)
}

func generateHTTPOptions(proxyURL, apiVersion string, customHeaders http.Header) *genai.HTTPOptions {
	httpOptions := &genai.HTTPOptions{
		BaseURL: proxyURL,
	}
	if customHeaders != nil {
		httpOptions.Headers = make(http.Header)
		for k, v := range customHeaders {
			httpOptions.Headers[k] = v
		}
	}
	if apiVersion != "" {
		httpOptions.APIVersion = apiVersion
	}
	return httpOptions
}

func backendType(isVertexAI bool) genai.Backend {
	if isVertexAI {
		return genai.BackendVertexAI
	}
	return genai.BackendGeminiAPI
}

// isEnabled returns true if the environment variable is set to "true" or "1".
func isEnabled(envVarName string) bool {
	val := os.Getenv(envVarName)
	return strings.ToLower(val) == "true" || val == "1"
}

func generateClientConfig(isVertexAI bool, backendType genai.Backend, httpOptions *genai.HTTPOptions, httpClient *http.Client) (*genai.ClientConfig, error) {
	clientConfig := &genai.ClientConfig{
		HTTPOptions: *httpOptions,
		Backend:     backendType,
	}

	project := os.Getenv(projectEnvVar)
	location := os.Getenv(locationEnvVar)
	if isVertexAI {
		if project == "" {
			return nil, fmt.Errorf("%s environment variable must be set", projectEnvVar)
		}
		if location == "" {
			return nil, fmt.Errorf("%s environment variable must be set", locationEnvVar)
		}
		clientConfig.Project = project
		clientConfig.Location = location
	}

	if httpClient != nil {
		clientConfig.HTTPClient = httpClient
	}

	return clientConfig, nil
}
