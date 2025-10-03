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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/genai"
)

// Artifacts implements Artifacts
type Artifacts struct {
	Service   artifact.Service
	AppName   string
	UserID    string
	SessionID string
}

func (a *Artifacts) Save(name string, data genai.Part) error {
	_, err := a.Service.Save(context.Background(), &artifact.SaveRequest{
		AppName:   a.AppName,
		UserID:    a.UserID,
		SessionID: a.SessionID,
		FileName:  name,
		Part:      &data,
	})
	return err
}

func (a *Artifacts) Load(name string) (genai.Part, error) {
	loadResponse, err := a.Service.Load(context.Background(), &artifact.LoadRequest{
		AppName:   a.AppName,
		UserID:    a.UserID,
		SessionID: a.SessionID,
		FileName:  name,
	})
	if err != nil {
		return genai.Part{}, err
	}
	return *loadResponse.Part, nil
}

func (a *Artifacts) LoadVersion(name string, version int) (genai.Part, error) {
	loadResponse, err := a.Service.Load(context.Background(), &artifact.LoadRequest{
		AppName:   a.AppName,
		UserID:    a.UserID,
		SessionID: a.SessionID,
		FileName:  name,
		Version:   int64(version),
	})
	if err != nil {
		return genai.Part{}, err
	}
	return *loadResponse.Part, nil
}

func (a *Artifacts) List() ([]string, error) {
	ListResponse, err := a.Service.List(context.Background(), &artifact.ListRequest{
		AppName:   a.AppName,
		UserID:    a.UserID,
		SessionID: a.SessionID,
	})
	if err != nil {
		return nil, err
	}
	return ListResponse.FileNames, nil
}

var _ agent.Artifacts = (*Artifacts)(nil)
