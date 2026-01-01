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

package openapitoolset

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/auth"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/internal/toolinternal/toolutils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

// RestApiTool is a tool that makes REST API calls.
type RestApiTool struct {
	name           string
	description    string
	method         string
	path           string
	baseURL        string
	parameters     []Parameter
	requestBody    *RequestBody
	authScheme     auth.AuthScheme
	authCredential *auth.AuthCredential
	httpClient     *http.Client
}

// sharedCredentialService is a package-level singleton for persistent token storage.
// This allows OAuth2 tokens to be cached across multiple requests within the same process.
var sharedCredentialService = auth.NewInMemoryCredentialService()

// newRestApiToolFromParsed creates a RestApiTool from a parsed operation.
func newRestApiToolFromParsed(parsed *ParsedOperation) *RestApiTool {
	return &RestApiTool{
		name:        parsed.Name,
		description: parsed.Description,
		method:      parsed.Method,
		path:        parsed.Path,
		baseURL:     parsed.BaseURL,
		parameters:  parsed.Parameters,
		requestBody: parsed.RequestBody,
		httpClient:  http.DefaultClient,
	}
}

// Name implements tool.Tool.
func (t *RestApiTool) Name() string {
	return t.name
}

// Description implements tool.Tool.
func (t *RestApiTool) Description() string {
	return t.description
}

// IsLongRunning implements tool.Tool.
func (t *RestApiTool) IsLongRunning() bool {
	return false
}

// Declaration returns the function declaration for the LLM.
func (t *RestApiTool) Declaration() *genai.FunctionDeclaration {
	// Build parameter schema from OpenAPI parameters
	properties := make(map[string]*genai.Schema)
	var required []string

	for _, p := range t.parameters {
		schema := convertSchemaToGenai(p.Schema)
		if schema == nil {
			schema = &genai.Schema{Type: genai.TypeString}
		}
		schema.Description = p.Description
		properties[p.Name] = schema
		if p.Required {
			required = append(required, p.Name)
		}
	}

	// Add request body as a parameter if present
	if t.requestBody != nil {
		for _, mt := range t.requestBody.Content {
			schema := convertSchemaToGenai(mt.Schema)
			if schema != nil {
				properties["body"] = schema
				if t.requestBody.Required {
					required = append(required, "body")
				}
				break // Use the first media type
			}
		}
	}

	return &genai.FunctionDeclaration{
		Name:        t.name,
		Description: t.description,
		Parameters: &genai.Schema{
			Type:       genai.TypeObject,
			Properties: properties,
			Required:   required,
		},
	}
}

// ProcessRequest implements toolinternal.RequestProcessor.
func (t *RestApiTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

// Run implements toolinternal.FunctionTool.
func (t *RestApiTool) Run(ctx tool.Context, args any) (map[string]any, error) {
	argsMap, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("args must be a map")
	}

	// Handle OAuth2 authentication flow
	if t.authScheme != nil && t.authCredential != nil {
		schemeType := t.authScheme.GetType()
		if schemeType == auth.SecuritySchemeTypeOAuth2 || schemeType == auth.SecuritySchemeTypeOpenIDConnect {
			// Create auth config for credential management
			authConfig, err := auth.NewAuthConfig(t.authScheme, t.authCredential)
			if err != nil {
				return nil, fmt.Errorf("failed to create auth config: %w", err)
			}

			// Check for existing credential from auth response
			authResponse, err := ctx.GetAuthResponse(authConfig)
			if err != nil {
				return nil, fmt.Errorf("failed to fetch auth response: %w", err)
			}
			if authResponse != nil {
				// User has completed OAuth flow - use the credential
				t.authCredential = authResponse
				// Save to shared credential service for persistence across requests
				authConfig.ExchangedAuthCredential = authResponse
				sharedCredentialService.SaveCredential(ctx, authConfig)
			} else {
				if t.authCredential.OAuth2 == nil || t.authCredential.OAuth2.AccessToken == "" {
					// No access token - need to get one
					manager := auth.NewCredentialManager(authConfig)
					// Use shared credential service for persistent token storage
					cred, err := manager.GetAuthCredential(ctx, func(key string) interface{} {
						val, _ := ctx.State().Get(key)
						return val
					}, sharedCredentialService)

					if err != nil {
						return nil, fmt.Errorf("failed to get auth credential: %w", err)
					}
					if cred == nil || (cred.OAuth2 != nil && cred.OAuth2.AccessToken == "") {
						// No credential available - request user authorization
						ctx.RequestCredential(authConfig)
						return map[string]any{
							"status":  "pending_authorization",
							"message": "User authorization required. Please complete the OAuth flow.",
						}, nil
					}
					// Update credential with exchanged token
					t.authCredential = cred
				}
			}
		}
	}

	// Build the request URL
	requestURL := t.buildURL(argsMap)

	// Build the request body
	var bodyReader io.Reader
	if body, ok := argsMap["body"]; ok && t.requestBody != nil {
		bodyBytes, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		bodyReader = bytes.NewReader(bodyBytes)
	}

	// Create the HTTP request
	req, err := http.NewRequestWithContext(ctx, t.method, requestURL, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	// Set headers
	if bodyReader != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Accept", "application/json")

	// Add auth headers
	for k, v := range t.getAuthHeaders() {
		req.Header.Set(k, v)
	}

	// Add header parameters
	for _, p := range t.parameters {
		if p.In == "header" {
			if val, ok := argsMap[p.Name]; ok {
				req.Header.Set(p.Name, fmt.Sprintf("%v", val))
			}
		}
	}

	// Execute the request
	resp, err := t.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read the response body
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Parse the response
	var result any
	if len(respBody) > 0 {
		contentType := resp.Header.Get("Content-Type")
		if strings.Contains(contentType, "application/json") {
			if err := json.Unmarshal(respBody, &result); err != nil {
				// Return as string if JSON parsing fails
				result = string(respBody)
			}
		} else {
			result = string(respBody)
		}
	}

	return map[string]any{
		"status_code": resp.StatusCode,
		"output":      result,
	}, nil
}

// buildURL builds the request URL with path and query parameters.
func (t *RestApiTool) buildURL(args map[string]any) string {
	path := t.path

	// Build query parameters
	query := url.Values{}

	for _, p := range t.parameters {
		val, ok := args[p.Name]
		if !ok {
			continue
		}

		valStr := fmt.Sprintf("%v", val)
		switch p.In {
		case "path":
			path = strings.ReplaceAll(path, "{"+p.Name+"}", url.PathEscape(valStr))
		case "query":
			query.Set(p.Name, valStr)
		}
	}

	result := t.baseURL + path
	if len(query) > 0 {
		result += "?" + query.Encode()
	}
	return result
}

// getAuthHeaders generates HTTP headers from the configured auth credential.
func (t *RestApiTool) getAuthHeaders() map[string]string {
	if t.authCredential == nil {
		return nil
	}

	headers := make(map[string]string)

	switch t.authCredential.AuthType {
	case auth.AuthCredentialTypeOAuth2:
		if t.authCredential.OAuth2 != nil && t.authCredential.OAuth2.AccessToken != "" {
			headers["Authorization"] = "Bearer " + t.authCredential.OAuth2.AccessToken
		}
	case auth.AuthCredentialTypeHTTP:
		if t.authCredential.HTTP != nil && t.authCredential.HTTP.Credentials != nil {
			creds := t.authCredential.HTTP.Credentials
			switch strings.ToLower(t.authCredential.HTTP.Scheme) {
			case "bearer":
				if creds.Token != "" {
					headers["Authorization"] = "Bearer " + creds.Token
				}
			case "basic":
				if creds.Username != "" && creds.Password != "" {
					encoded := base64.StdEncoding.EncodeToString(
						[]byte(creds.Username + ":" + creds.Password),
					)
					headers["Authorization"] = "Basic " + encoded
				}
			default:
				if creds.Token != "" {
					headers["Authorization"] = t.authCredential.HTTP.Scheme + " " + creds.Token
				}
			}
		}
	case auth.AuthCredentialTypeAPIKey:
		if t.authCredential.APIKey != "" && t.authScheme != nil {
			if apiKeyScheme, ok := t.authScheme.(*auth.APIKeyScheme); ok {
				switch apiKeyScheme.In {
				case auth.APIKeyInHeader:
					headers[apiKeyScheme.Name] = t.authCredential.APIKey
				}
			}
		}
	}

	if len(headers) == 0 {
		return nil
	}
	return headers
}

// ConfigureAuthScheme sets the auth scheme for this tool.
func (t *RestApiTool) ConfigureAuthScheme(scheme auth.AuthScheme) {
	t.authScheme = scheme
}

// ConfigureAuthCredential sets the auth credential for this tool.
func (t *RestApiTool) ConfigureAuthCredential(cred *auth.AuthCredential) {
	t.authCredential = cred
}

// convertSchemaToGenai converts an OpenAPI schema to a genai.Schema.
func convertSchemaToGenai(schema map[string]any) *genai.Schema {
	if schema == nil {
		return nil
	}

	result := &genai.Schema{}

	// Get type
	if typeStr, ok := schema["type"].(string); ok {
		switch typeStr {
		case "string":
			result.Type = genai.TypeString
		case "integer":
			result.Type = genai.TypeInteger
		case "number":
			result.Type = genai.TypeNumber
		case "boolean":
			result.Type = genai.TypeBoolean
		case "array":
			result.Type = genai.TypeArray
			if items, ok := schema["items"].(map[string]any); ok {
				result.Items = convertSchemaToGenai(items)
			}
		case "object":
			result.Type = genai.TypeObject
			if props, ok := schema["properties"].(map[string]any); ok {
				result.Properties = make(map[string]*genai.Schema)
				for name, propSchema := range props {
					if ps, ok := propSchema.(map[string]any); ok {
						result.Properties[name] = convertSchemaToGenai(ps)
					}
				}
			}
		}
	}

	// Get description
	if desc, ok := schema["description"].(string); ok {
		result.Description = desc
	}

	// Get enum values
	if enum, ok := schema["enum"].([]any); ok {
		for _, e := range enum {
			if s, ok := e.(string); ok {
				result.Enum = append(result.Enum, s)
			}
		}
	}

	return result
}

var (
	_ toolinternal.FunctionTool     = (*RestApiTool)(nil)
	_ toolinternal.RequestProcessor = (*RestApiTool)(nil)
)
