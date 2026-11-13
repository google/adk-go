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

package artifact

import (
	"context"
	"fmt"
	"io/fs"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
)

// Artifacts implements Artifacts
type Artifacts struct {
	Service   artifact.Service
	AppName   string
	UserID    string
	SessionID string
}

func (a *Artifacts) Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	return a.Service.Save(ctx, &artifact.SaveRequest{
		AppName:   a.AppName,
		UserID:    a.UserID,
		SessionID: a.SessionID,
		FileName:  name,
		Part:      data,
	})
}

func (a *Artifacts) Load(ctx context.Context, name string) (*artifact.LoadResponse, error) {
	return a.Service.Load(ctx, &artifact.LoadRequest{
		AppName:   a.AppName,
		UserID:    a.UserID,
		SessionID: a.SessionID,
		FileName:  name,
	})
}

func (a *Artifacts) LoadVersion(ctx context.Context, name string, version int) (*artifact.LoadResponse, error) {
	return a.Service.Load(ctx, &artifact.LoadRequest{
		AppName:   a.AppName,
		UserID:    a.UserID,
		SessionID: a.SessionID,
		FileName:  name,
		Version:   int64(version),
	})
}

func (a *Artifacts) List(ctx context.Context) (*artifact.ListResponse, error) {
	return a.Service.List(ctx, &artifact.ListRequest{
		AppName:   a.AppName,
		UserID:    a.UserID,
		SessionID: a.SessionID,
	})
}

// ArtifactVersion returns the record of version of the named artifact in a's
// session, or of its latest version when version is nil.
//
// When a's service forwards to the artifacts of a parent context that has no
// artifact service, through any number of nested AgentTools, it returns that
// parent's error first, whatever the version. A non-nil version of 0 or less
// returns an error wrapping [fs.ErrNotExist] without calling the service.
// Otherwise it returns the service's error unchanged; a missing artifact or
// version is an error wrapping [fs.ErrNotExist].
func (a *Artifacts) ArtifactVersion(ctx context.Context, name string, version *int64) (*artifact.ArtifactVersion, error) {
	if err := forwardedServiceErr(a.Service); err != nil {
		return nil, err
	}
	req := &artifact.GetArtifactVersionRequest{
		AppName:   a.AppName,
		UserID:    a.UserID,
		SessionID: a.SessionID,
		FileName:  name,
	}
	if version != nil {
		// The service reads version 0 as the latest; versions start at 1.
		if *version <= 0 {
			return nil, fmt.Errorf("artifact not found: version is not positive: %w", fs.ErrNotExist)
		}
		req.Version = *version
	}
	resp, err := a.Service.GetArtifactVersion(ctx, req)
	if err != nil {
		return nil, err
	}
	return resp.ArtifactVersion, nil
}

var _ agent.Artifacts = (*Artifacts)(nil)
