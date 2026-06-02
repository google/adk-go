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

package vertexai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"golang.org/x/oauth2/google"
)

var errSkillNotFound = errors.New("skill not found")

type Skill struct {
	Name             string            `json:"name"`
	Description      string            `json:"description"`
	License          string            `json:"license,omitempty"`
	Compatibility    string            `json:"compatibility,omitempty"`
	Metadata         map[string]string `json:"metadata,omitempty"`
	AllowedTools     []string          `json:"allowed-tools,omitempty"`
	ZippedFilesystem string            `json:"zippedFilesystem"`
}

type SearchSkillsResponse struct {
	Skills []struct {
		Name        string `json:"skillName"`
		Description string `json:"description"`
	} `json:"retrievedSkills"`
}

type ErrorResponse struct {
	Error struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

type vertexAIClient struct {
	config  vertexAIClientConfig
	client  *http.Client
	baseUri string
}

type vertexAIClientConfig struct {
	Location  string
	ProjectID string
}

func newVertexAIClient(ctx context.Context, config *vertexAIClientConfig) (*vertexAIClient, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}
	c, err := google.DefaultClient(ctx, "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return nil, err
	}
	return &vertexAIClient{
		config:  *config,
		client:  c,
		baseUri: fmt.Sprintf("https://%[1]s-aiplatform.googleapis.com/v1beta1/projects/%[2]s/locations/%[1]s/skills", config.Location, config.ProjectID),
	}, nil
}

func (v *vertexAIClient) getSkill(ctx context.Context, skillId string) (*Skill, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, v.baseUri+"/"+skillId, nil)
	if err != nil {
		return nil, err
	}
	resp, err := v.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return nil, v.handleAPIError(resp)
	}

	var skill *Skill
	err = json.NewDecoder(resp.Body).Decode(&skill)
	if err != nil {
		return nil, fmt.Errorf("failed to decode json: %w", err)
	}
	return skill, nil
}

func (v *vertexAIClient) searchSkills(ctx context.Context, query string) (*SearchSkillsResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, v.baseUri+":retrieve", nil)
	if err != nil {
		return nil, err
	}
	q := req.URL.Query()
	q.Add("query", query)
	req.URL.RawQuery = q.Encode()

	resp, err := v.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return nil, v.handleAPIError(resp)
	}

	var response SearchSkillsResponse
	err = json.NewDecoder(resp.Body).Decode(&response)
	if err != nil {
		return nil, fmt.Errorf("failed to decode json: %w", err)
	}
	return &response, nil
}

func (v *vertexAIClient) handleAPIError(resp *http.Response) error {
	if resp.StatusCode == http.StatusNotFound {
		return errSkillNotFound
	}
	var errRes ErrorResponse
	if err := json.NewDecoder(resp.Body).Decode(&errRes); err != nil {
		return fmt.Errorf("request returned status %d, but failed to decode error json: %w", resp.StatusCode, err)
	}
	return fmt.Errorf("skills registry api failed with error: %s (%d - %s)", errRes.Error.Message, errRes.Error.Code, errRes.Error.Status)
}
