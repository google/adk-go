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

package runner

import (
	"context"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifactservice"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// artifacts implements Artifacts
type artifacts struct {
	service artifactservice.Service
	id      session.ID
}

func (a *artifacts) Save(name string, data genai.Part) error {
	_, err := a.service.Save(context.Background(), &artifactservice.SaveRequest{
		AppName:   a.id.AppName,
		UserID:    a.id.UserID,
		SessionID: a.id.SessionID,
		FileName:  name,
		Part:      &data,
	})
	return err
}

func (a *artifacts) Load(name string) (genai.Part, error) {
	loadResponse, err := a.service.Load(context.Background(), &artifactservice.LoadRequest{
		AppName:   a.id.AppName,
		UserID:    a.id.UserID,
		SessionID: a.id.SessionID,
		FileName:  name,
	})
	return *loadResponse.Part, err
}

func (a *artifacts) LoadVersion(name string, version int) (genai.Part, error) {
	loadResponse, err := a.service.Load(context.Background(), &artifactservice.LoadRequest{
		AppName:   a.id.AppName,
		UserID:    a.id.UserID,
		SessionID: a.id.SessionID,
		FileName:  name,
		Version:   int64(version),
	})
	return *loadResponse.Part, err
}

func (a *artifacts) List() ([]string, error) {
	ListResponse, err := a.service.List(context.Background(), &artifactservice.ListRequest{
		AppName:   a.id.AppName,
		UserID:    a.id.UserID,
		SessionID: a.id.SessionID,
	})
	return ListResponse.FileNames, err
}

var _ agent.Artifacts = (*artifacts)(nil)
