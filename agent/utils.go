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
	"strings"

	"github.com/google/adk-go"
	"github.com/google/uuid"
	"google.golang.org/genai"
)

const afFunctionCallIDPrefix = "adk-"

// populateClientFunctionCallID sets the function call ID field if it is empty.
// Since the ID field is optional, some models don't fill the field, but
// the LLMAgent depends on the IDs to map FunctionCall and FunctionResponse events
// in the event stream.
func populateClientFunctionCallID(c *genai.Content) {
	for _, fn := range functionCalls(c) {
		if fn.ID == "" {
			fn.ID = afFunctionCallIDPrefix + uuid.NewString()
		}
	}
}

// removeClientFunctionCallID removes the function call ID field that was set
// by populateClientFunctionCallID. This is necessary when FunctionCall or
// FunctionResponse are sent back to the model.
func removeClientFunctionCallID(c *genai.Content) {
	for _, fn := range functionCalls(c) {
		if strings.HasPrefix(fn.ID, afFunctionCallIDPrefix) {
			fn.ID = ""
		}
	}
	for _, fn := range functionResponses(c) {
		if strings.HasPrefix(fn.ID, afFunctionCallIDPrefix) {
			fn.ID = ""
		}
	}
}

//
// Belows are useful utilities that help working with genai.Content
// included in adk.Event.
// TODO: Move them to a separate internal utility package.
// TODO: Use generics.

// functionCalls extracts all FunctionCall parts from the content.
func functionCalls(c *genai.Content) (ret []*genai.FunctionCall) {
	if c == nil {
		return nil
	}
	for _, p := range c.Parts {
		if p.FunctionCall != nil {
			ret = append(ret, p.FunctionCall)
		}
	}
	return ret
}

// functionResponses extracts all FunctionResponse parts from the content.
func functionResponses(c *genai.Content) (ret []*genai.FunctionResponse) {
	if c == nil {
		return nil
	}
	for _, p := range c.Parts {
		if p.FunctionResponse != nil {
			ret = append(ret, p.FunctionResponse)
		}
	}
	return ret
}

// textParts extracts all Text parts from the content.
func textParts(c *genai.Content) (ret []string) {
	if c == nil {
		return nil
	}
	for _, p := range c.Parts {
		if p.Text != "" {
			ret = append(ret, p.Text)
		}
	}
	return ret
}

// functionDecls extracts all Function declarations from the GenerateContentConfig.
func functionDecls(c *genai.GenerateContentConfig) (ret []*genai.FunctionDeclaration) {
	if c == nil {
		return nil
	}
	for _, t := range c.Tools {
		ret = append(ret, t.FunctionDeclarations...)
	}
	return ret
}

// content is a convenience function that returns the genai.Content
// in the event.
func content(ev *adk.Event) *genai.Content {
	if ev == nil || ev.LLMResponse == nil {
		return nil
	}
	return ev.LLMResponse.Content
}

// rootAgent returns the root of the agent tree.
func rootAgent(agent adk.Agent) adk.Agent {
	findParent := func(agent adk.Agent) adk.Agent {
		// this test is because ParentAgent/SubAgents are implemented only in LLMAgent.
		// TODO: Parent/SubAgents should be part of Agent interface. Then, this is not needed.
		a := asLLMAgent(agent)
		if a == nil {
			return nil
		}
		if p := asLLMAgent(a.ParentAgent); p != nil {
			return p
		}
		return nil
	}
	current := agent
	for {
		parent := findParent(current)
		if parent == nil {
			return current
		}
		current = parent
	}
}
