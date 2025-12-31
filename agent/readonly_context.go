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

	"google.golang.org/genai"

	"google.golang.org/adk/session"
)

// NewReadonlyContext creates a ReadonlyContext from an InvocationContext
func NewReadonlyContext(ctx InvocationContext) ReadonlyContext {
	return &readonlyContext{
		Context:           ctx,
		invocationContext: ctx,
	}
}

func GetInvocationContextFromReadonly(ctx ReadonlyContext) InvocationContext {
	readonlyCtx, ok := ctx.(*readonlyContext)
	if !ok {
		return nil
	}
	return readonlyCtx.invocationContext
}

// readonlyContext implements ReadonlyContext
type readonlyContext struct {
	context.Context
	invocationContext InvocationContext
}

func (r *readonlyContext) UserContent() *genai.Content {
	return r.invocationContext.UserContent()
}

func (r *readonlyContext) InvocationID() string {
	return r.invocationContext.InvocationID()
}

func (r *readonlyContext) AgentName() string {
	return r.invocationContext.Agent().Name()
}

func (r *readonlyContext) ReadonlyState() session.ReadonlyState {
	return r.invocationContext.Session().State()
}

func (r *readonlyContext) UserID() string {
	return r.invocationContext.Session().UserID()
}

func (r *readonlyContext) AppName() string {
	return r.invocationContext.Session().AppName()
}

func (r *readonlyContext) SessionID() string {
	return r.invocationContext.Session().ID()
}

func (r *readonlyContext) Branch() string {
	return r.invocationContext.Branch()
}
