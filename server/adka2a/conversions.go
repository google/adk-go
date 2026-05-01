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

package adka2a

import (
	"context"
	"fmt"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/log"
	v2a2a "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2acompat/a2av0"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	v2 "google.golang.org/adk/server/adka2a/v2"
	"google.golang.org/adk/session"
)

// BuildAgentSkills attempts to create a list of [a2a.AgentSkill]s based on agent descriptions and types.
// This information can be used in [a2a.AgentCard] to help clients understand agent capabilities.
func BuildAgentSkills(agent agent.Agent) []a2a.AgentSkill {
	v1Skills := v2.BuildAgentSkills(agent)
	card := a2av0.FromV1AgentCard(&v2a2a.AgentCard{Skills: v1Skills})
	return card.Skills
}

// ToA2AMetaKey adds a prefix used to differentiate ADK-related values stored in Metadata an A2A event.
func ToA2AMetaKey(key string) string {
	return v2.ToA2AMetaKey(key)
}

// ToADKMetaKey adds a prefix used to differentiate A2A-related values stored in custom metadata of an ADK session event.
func ToADKMetaKey(key string) string {
	return v2.ToADKMetaKey(key)
}

// ToCustomMetadata creates ADK custom metadata from A2A task and context IDs.
func ToCustomMetadata(taskID a2a.TaskID, contextID string) map[string]any {
	return v2.ToCustomMetadata(v2a2a.TaskID(taskID), contextID)
}

// GetA2ATaskInfo returns A2A task and context IDs if they are present in session event custom metadata.
func GetA2ATaskInfo(event *session.Event) (a2a.TaskID, string) {
	taskID, contextID := v2.GetA2ATaskInfo(event)
	return a2a.TaskID(taskID), contextID
}

// NewRemoteAgentEvent create a new Event authored by the agent running in the provided invocation context.
func NewRemoteAgentEvent(ctx agent.InvocationContext) *session.Event {
	return v2.NewRemoteAgentEvent(ctx)
}

// EventToMessage converts the provided session event to A2A message.
func EventToMessage(event *session.Event) (*a2a.Message, error) {
	v1Msg, err := v2.EventToMessage(event)
	if err != nil {
		return nil, err
	}
	return a2av0.FromV1Message(v1Msg), nil
}

// ToSessionEvent converts the provided a2a event to session event authored by the agent running in the provided invocation context.
func ToSessionEvent(ctx agent.InvocationContext, event a2a.Event) (*session.Event, error) {
	v1Event, err := a2av0.ToV1Event(event)
	if err != nil {
		return nil, fmt.Errorf("a2a event conversion failed: %w", err)
	}
	return v2.ToSessionEvent(ctx, v1Event)
}

// IsPartial takes metadata of an A2A object (eg. a2a.Part, a2a.Artifact) and returns true if
// it was marked as partial based on the ADK partial flag set on the original ADK object.
func IsPartial(meta map[string]any) bool {
	return v2.IsPartial(meta)
}

// IsPartialFlagSet takes metadata of an A2A object (eg. a2a.Part, a2a.Artifact) and returns true if
// the ADK partial flag was set on it.
func IsPartialFlagSet(meta map[string]any) bool {
	return v2.IsPartialFlagSet(meta)
}

// ToA2APart converts the provided genai part to A2A equivalent. Long running tool IDs are used for attaching metadata to
// the relevant data parts.
func ToA2APart(part *genai.Part, longRunningToolIDs []string) (a2a.Part, error) {
	v1p, err := v2.ToA2APart(part, longRunningToolIDs)
	if err != nil {
		return nil, err
	}
	return a2av0.FromV1Part(v1p), nil
}

// ToA2AParts converts the provided genai parts to A2A equivalents. Long running tool IDs are used for attaching metadata to
// the relevant data parts.
func ToA2AParts(parts []*genai.Part, longRunningToolIDs []string) ([]a2a.Part, error) {
	v1ps, err := v2.ToA2AParts(parts, longRunningToolIDs)
	if err != nil {
		return nil, err
	}
	result := make([]a2a.Part, len(v1ps))
	for i, v1p := range v1ps {
		result[i] = a2av0.FromV1Part(v1p)
	}
	return result, nil
}

// ToGenAIPart converts the provided A2A part to a genai equivalent.
func ToGenAIPart(part a2a.Part) (*genai.Part, error) {
	return v2.ToGenAIPart(a2av0.ToV1Part(part))
}

// ToGenAIParts converts the provided A2A parts to genai equivalents.
func ToGenAIParts(parts []a2a.Part) ([]*genai.Part, error) {
	v1ps := make([]*v2a2a.Part, len(parts))
	for i, p := range parts {
		v1ps[i] = a2av0.ToV1Part(p)
	}
	return v2.ToGenAIParts(v1ps)
}

// WithoutPartialArtifacts returns a slice of artifacts without partial artifacts.
// Partial artifacts are usually discarded (contain no parts) after agent invocation is finished.
func WithoutPartialArtifacts(artifacts []*a2a.Artifact) []*a2a.Artifact {
	v1Artifacts := make([]*v2a2a.Artifact, 0, len(artifacts))
	for _, a := range artifacts {
		v1a, err := a2av0.ToV1Artifact(a)
		if err != nil {
			log.Warn(context.Background(), "artifact skipped, because conversion failed", "cause", err)
			continue
		}
		v1Artifacts = append(v1Artifacts, v1a)
	}

	filtered := v2.WithoutPartialArtifacts(v1Artifacts)

	result := make([]*a2a.Artifact, 0, len(filtered))
	for _, f := range filtered {
		result = append(result, a2av0.FromV1Artifact(f))
	}
	return result
}
