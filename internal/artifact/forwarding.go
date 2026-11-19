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

package artifact

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
)

var errArtifactServiceNotSet = errors.New("parent artifact service is not set")

// NewForwardingService forwards artifact operations to parent and records saves
// in its event delta. Operations fail if parent is nil.
// Delete, Versions and GetArtifactVersion return errors.ErrUnsupported when
// the parent's backing service is inaccessible.
func NewForwardingService(parent agent.Artifacts) artifact.Service {
	return &forwardingService{parent: parent}
}

type forwardingService struct {
	parent agent.Artifacts
	// Parallel child agents share the parent's event delta, which is a plain map.
	saveMu sync.Mutex
}

func (s *forwardingService) Save(ctx context.Context, req *artifact.SaveRequest) (*artifact.SaveResponse, error) {
	if s.parent == nil {
		return nil, errArtifactServiceNotSet
	}
	s.saveMu.Lock()
	defer s.saveMu.Unlock()
	return s.parent.Save(ctx, req.FileName, req.Part)
}

func (s *forwardingService) Load(ctx context.Context, req *artifact.LoadRequest) (*artifact.LoadResponse, error) {
	if s.parent == nil {
		return nil, errArtifactServiceNotSet
	}
	if req.Version == 0 {
		return s.parent.Load(ctx, req.FileName)
	}
	return s.parent.LoadVersion(ctx, req.FileName, int(req.Version))
}

func (s *forwardingService) List(ctx context.Context, _ *artifact.ListRequest) (*artifact.ListResponse, error) {
	if s.parent == nil {
		return nil, errArtifactServiceNotSet
	}
	return s.parent.List(ctx)
}

func (s *forwardingService) Delete(ctx context.Context, req *artifact.DeleteRequest) error {
	parent, err := s.parentStorage()
	if err != nil {
		return err
	}
	parentReq := *req
	parentReq.AppName, parentReq.UserID, parentReq.SessionID = parent.AppName, parent.UserID, parent.SessionID
	return parent.Service.Delete(ctx, &parentReq)
}

func (s *forwardingService) Versions(ctx context.Context, req *artifact.VersionsRequest) (*artifact.VersionsResponse, error) {
	parent, err := s.parentStorage()
	if err != nil {
		return nil, err
	}
	parentReq := *req
	parentReq.AppName, parentReq.UserID, parentReq.SessionID = parent.AppName, parent.UserID, parent.SessionID
	return parent.Service.Versions(ctx, &parentReq)
}

func (s *forwardingService) GetArtifactVersion(ctx context.Context, req *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	parent, err := s.parentStorage()
	if err != nil {
		return nil, err
	}
	parentReq := *req
	parentReq.AppName, parentReq.UserID, parentReq.SessionID = parent.AppName, parent.UserID, parent.SessionID
	return parent.Service.GetArtifactVersion(ctx, &parentReq)
}

func (s *forwardingService) parentStorage() (*Artifacts, error) {
	if s.parent == nil {
		return nil, errArtifactServiceNotSet
	}
	parent := s.parent
	for {
		switch a := parent.(type) {
		case *Artifacts:
			return a, nil
		case interface{ Unwrap() agent.Artifacts }:
			parent = a.Unwrap()
		default:
			return nil, fmt.Errorf("parent artifacts do not expose a backing service: %w", errors.ErrUnsupported)
		}
	}
}
