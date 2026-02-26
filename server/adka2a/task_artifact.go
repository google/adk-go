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

package adka2a

import (
	"maps"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"

	"google.golang.org/adk/session"
)

type artifactMaker struct {
	reqCtx                   *a2asrv.RequestContext
	lastAgentPartialArtifact map[string]a2a.ArtifactID
}

func newArtifactMaker(reqCtx *a2asrv.RequestContext) *artifactMaker {
	return &artifactMaker{
		reqCtx:                   reqCtx,
		lastAgentPartialArtifact: make(map[string]a2a.ArtifactID),
	}
}

var _ eventToArtifactTransform = (*artifactMaker)(nil)

func (m *artifactMaker) transform(event *session.Event, parts []a2a.Part, meta map[string]any) (*a2a.TaskArtifactUpdateEvent, error) {
	result := a2a.NewArtifactEvent(m.reqCtx, parts...)

	if artifactID, ok := m.lastAgentPartialArtifact[event.Author]; ok {
		result.Artifact.ID = artifactID
		// continue accumulating if partial, otherwise replace contents
		result.Append = event.Partial
	}
	result.LastChunk = !event.Partial

	if event.Partial {
		m.lastAgentPartialArtifact[event.Author] = result.Artifact.ID
	} else {
		delete(m.lastAgentPartialArtifact, event.Author)
	}

	if len(meta) > 0 {
		if result.Artifact.Metadata == nil {
			result.Artifact.Metadata = make(map[string]any)
		}
		maps.Copy(result.Artifact.Metadata, meta)
	}

	return result, nil
}

func (m *artifactMaker) makeFinalUpdate() *a2a.TaskArtifactUpdateEvent {
	return nil
}

type legacyArtifactMaker struct {
	reqCtx *a2asrv.RequestContext

	// responseID is created once the first TaskArtifactUpdateEvent is sent. Used for subsequent artifact updates.
	responseID a2a.ArtifactID
	// partialResponseID is created once the first TaskArtifactUpdateEvent created from a partial ADK event is sent.
	// Partial updates are not saved in the ADK session store. There is no concept of a partial event in A2A so instead
	// we're updating an "ephemeral" artifact while an agent is running. The artifact gets reset at the end of the
	// invocation effectively erasing its parts.
	partialResponseID a2a.ArtifactID
}

func newLegacyArtifactMaker(reqCtx *a2asrv.RequestContext) *legacyArtifactMaker {
	return &legacyArtifactMaker{
		reqCtx: reqCtx,
	}
}

var _ eventToArtifactTransform = (*legacyArtifactMaker)(nil)

func (p *legacyArtifactMaker) transform(event *session.Event, parts []a2a.Part, meta map[string]any) (*a2a.TaskArtifactUpdateEvent, error) {
	var result *a2a.TaskArtifactUpdateEvent
	if event.Partial {
		result = newLegacyPartialArtifactUpdate(p.reqCtx, p.partialResponseID, parts)
		p.partialResponseID = result.Artifact.ID
	} else {
		result = newLegacyArtifactUpdate(p.reqCtx, p.responseID, parts)
		p.responseID = result.Artifact.ID
	}
	if len(meta) > 0 {
		maps.Copy(result.Metadata, meta)
	}
	return result, nil
}

func (p *legacyArtifactMaker) makeFinalUpdate() *a2a.TaskArtifactUpdateEvent {
	// We could also send a LastChunk: true event for the main (non-partial) artifact,
	// but there's currently no special handling for it and not all A2A SDK (eg. Java)
	// implementations allow empty-part artifact updates.
	if p.partialResponseID == "" {
		return nil
	}
	ev := newLegacyPartialArtifactUpdate(p.reqCtx, p.partialResponseID, []a2a.Part{a2a.DataPart{Data: map[string]any{}}})
	ev.LastChunk = true
	return ev
}

func newLegacyArtifactUpdate(task a2a.TaskInfoProvider, id a2a.ArtifactID, parts []a2a.Part) *a2a.TaskArtifactUpdateEvent {
	var result *a2a.TaskArtifactUpdateEvent
	if id == "" {
		result = a2a.NewArtifactEvent(task, parts...)
	} else {
		result = a2a.NewArtifactUpdateEvent(task, id, parts...)
	}
	// Explicitely mark and Artifact update as non-partial ADK event so that consumer side
	// does not run its own aggregation logic.
	result.Metadata = map[string]any{metadataPartialKey: false}
	return result
}

func newLegacyPartialArtifactUpdate(task a2a.TaskInfoProvider, artifactID a2a.ArtifactID, parts []a2a.Part) *a2a.TaskArtifactUpdateEvent {
	ev := newLegacyArtifactUpdate(task, artifactID, parts)
	updatePartsMetadata(parts, map[string]any{metadataPartialKey: true})
	if ev.Artifact.Metadata == nil {
		ev.Artifact.Metadata = map[string]any{metadataPartialKey: true}
	} else {
		ev.Artifact.Metadata[metadataPartialKey] = true
	}
	ev.Metadata[metadataPartialKey] = true
	ev.Append = false // discard partial events
	return ev
}
