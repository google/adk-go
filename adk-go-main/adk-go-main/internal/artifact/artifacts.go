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

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
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

var _ agent.Artifacts = (*Artifacts)(nil)
