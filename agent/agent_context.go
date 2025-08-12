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

package agent

import (
	"context"

	"github.com/google/uuid"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

type agentContext struct {
	context.Context
	cancel context.CancelFunc

	invocationID string
	agent        Agent
	session      session.Session
	userContent  *genai.Content
	branch       string
}

// TODO: see if needed or possible to make internal
func NewContext(ctx context.Context, agent Agent, userContent *genai.Content, session session.Session, branch string) *agentContext {
	ctx, cancel := context.WithCancel(ctx)

	return &agentContext{
		Context: ctx,
		cancel:  cancel,

		invocationID: "e-" + uuid.NewString(),
		agent:        agent,
		session:      session,
		userContent:  userContent,
		branch:       branch,
	}
}

func (a *agentContext) UserContent() *genai.Content {
	return a.userContent
}

func (a *agentContext) InvocationID() string {
	return a.invocationID
}

func (a *agentContext) Branch() string {
	return a.branch
}

func (a *agentContext) Agent() Agent {
	return a.agent
}

func (a *agentContext) Session() session.Session {
	return a.session
}

func (*agentContext) Artifacts() Artifacts {
	return nil
}

func (*agentContext) Report(*session.Event) {

}

func (a *agentContext) End() {
	a.cancel()
}

func (a *agentContext) Ended() bool {
	return a.Context.Err() != nil
}

var _ Context = (*agentContext)(nil)
