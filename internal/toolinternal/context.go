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

package toolinternal

import (
	"context"
	"errors"
	"fmt"

	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/auth"
	contextinternal "google.golang.org/adk/internal/context"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

type internalArtifacts struct {
	agent.Artifacts
	eventActions *session.EventActions
}

func (ia *internalArtifacts) Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	resp, err := ia.Artifacts.Save(ctx, name, data)
	if err != nil {
		return resp, err
	}
	if ia.eventActions != nil {
		if ia.eventActions.ArtifactDelta == nil {
			ia.eventActions.ArtifactDelta = make(map[string]int64)
		}
		// TODO: RWLock, check the version stored is newer in case multiple tools save the same file.
		ia.eventActions.ArtifactDelta[name] = resp.Version
	}
	return resp, nil
}

func NewToolContext(ctx agent.InvocationContext, functionCallID string, actions *session.EventActions) tool.Context {
	if functionCallID == "" {
		functionCallID = uuid.NewString()
	}
	if actions == nil {
		actions = &session.EventActions{
			StateDelta:           make(map[string]any),
			RequestedAuthConfigs: make(map[string]*auth.AuthConfig),
		}
	}
	if actions.StateDelta == nil {
		actions.StateDelta = make(map[string]any)
	}
	if actions.RequestedAuthConfigs == nil {
		actions.RequestedAuthConfigs = make(map[string]*auth.AuthConfig)
	}
	cbCtx := contextinternal.NewCallbackContextWithDelta(ctx, actions.StateDelta)

	return &toolContext{
		CallbackContext:   cbCtx,
		invocationContext: ctx,
		functionCallID:    functionCallID,
		eventActions:      actions,
		artifacts: &internalArtifacts{
			Artifacts:    ctx.Artifacts(),
			eventActions: actions,
		},
	}
}

type toolContext struct {
	agent.CallbackContext
	invocationContext agent.InvocationContext
	functionCallID    string
	eventActions      *session.EventActions
	artifacts         *internalArtifacts
}

func (c *toolContext) Artifacts() agent.Artifacts {
	return c.artifacts
}

func (c *toolContext) FunctionCallID() string {
	return c.functionCallID
}

func (c *toolContext) Actions() *session.EventActions {
	return c.eventActions
}

func (c *toolContext) AgentName() string {
	return c.invocationContext.Agent().Name()
}

func (c *toolContext) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	return c.invocationContext.Memory().Search(ctx, query)
}

// RequestCredential requests user authorization for OAuth2.
// The auth config will be included in the event's RequestedAuthConfigs,
// which is converted to adk_request_credential function calls by GenerateAuthEvent.
func (c *toolContext) RequestCredential(config *auth.AuthConfig) error {

	if config == nil {
		return fmt.Errorf("auth config is nil")
	}

	// Generate auth request with auth_uri
	handler := auth.NewAuthHandler(config)
	authRequest, err := handler.GenerateAuthRequest()
	if err != nil {
		return fmt.Errorf("generate auth request: %w", err)
	}
	if authRequest == nil {
		return fmt.Errorf("generate auth request: empty result")
	}

	// Add to RequestedAuthConfigs keyed by function call ID
	c.eventActions.RequestedAuthConfigs[c.functionCallID] = authRequest
	return nil
}

// GetAuthResponse retrieves the auth response from session state.
// Returns nil if no auth response is available.
func (c *toolContext) GetAuthResponse(config *auth.AuthConfig) (*auth.AuthCredential, error) {
	if config == nil {
		return nil, fmt.Errorf("auth config is nil")
	}
	key := session.KeyPrefixTemp + config.CredentialKey

	val, err := c.invocationContext.Session().State().Get(key)
	if err != nil {
		if errors.Is(err, session.ErrStateKeyNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("get auth response: %w", err)
	}
	if val == nil {
		return nil, nil
	}

	cred, ok := val.(*auth.AuthCredential)
	if !ok {
		return nil, fmt.Errorf("unexpected auth response type %T", val)
	}

	return cred, nil
}

// CredentialService returns the credential service for persistent storage.
// Returns nil as toolContext does not have a default credential service.
// The InvocationContext or runner should provide a credential service if needed.
func (c *toolContext) CredentialService() auth.CredentialService {
	return nil
}
