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

package agent

import (
	"context"

	"google.golang.org/genai"
)

type CommonContextDelta struct {
	InvocationContextDelta *InvocationContextDelta
}

type InvocationContextDelta struct {
	Context     *context.Context
	UserContent **genai.Content
	Agent       *Agent
}

func (c *commonContext) Apply(d *CommonContextDelta) Context {
	if d == nil {
		return c
	}
	res := *c
	res.invocationContext = res.invocationContext.ApplyICDelta(d.InvocationContextDelta)

	if d.InvocationContextDelta != nil {
		if d.InvocationContextDelta.Context != nil {
			res.Context = *d.InvocationContextDelta.Context
		}
	}

	return &res
}

func (c *commonContext) ApplyICDelta(d *InvocationContextDelta) InvocationContext {
	if d == nil {
		return c
	}
	res := *c
	res.invocationContext = res.invocationContext.ApplyICDelta(d)
	return &res
}
