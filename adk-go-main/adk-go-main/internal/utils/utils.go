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

package utils

import (
	"strings"

	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// TODO: split in proper files/packages.

const afFunctionCallIDPrefix = "adk-"

// PopulateClientFunctionCallID sets the function call ID field if it is empty.
// Since the ID field is optional, some models don't fill the field, but
// the LLMAgent depends on the IDs to map FunctionCall and FunctionResponse events
// in the event stream.
func PopulateClientFunctionCallID(c *genai.Content) {
	for _, fn := range FunctionCalls(c) {
		if fn.ID == "" {
			fn.ID = afFunctionCallIDPrefix + uuid.NewString()
		}
	}
}

// RemoveClientFunctionCallID removes the function call ID field that was set
// by populateClientFunctionCallID. This is necessary when FunctionCall or
// FunctionResponse are sent back to the model.
func RemoveClientFunctionCallID(c *genai.Content) {
	for _, fn := range FunctionCalls(c) {
		if strings.HasPrefix(fn.ID, afFunctionCallIDPrefix) {
			fn.ID = ""
		}
	}
	for _, fn := range FunctionResponses(c) {
		if strings.HasPrefix(fn.ID, afFunctionCallIDPrefix) {
			fn.ID = ""
		}
	}
}

// Content is a convenience function that returns the genai.Content
// in the event.
func Content(ev *session.Event) *genai.Content {
	if ev == nil {
		return nil
	}
	return ev.LLMResponse.Content
}

// Belows are useful utilities that help working with genai.Content
// included in types.Event.
// TODO: Use generics.
// FunctionCalls extracts all FunctionCall parts from the content.
func FunctionCalls(c *genai.Content) (ret []*genai.FunctionCall) {
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

// FunctionResponses extracts all FunctionResponse parts from the content.
func FunctionResponses(c *genai.Content) (ret []*genai.FunctionResponse) {
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

// TextParts extracts all Text parts from the content.
func TextParts(c *genai.Content) (ret []string) {
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

// FunctionDecls extracts all Function declarations from the GenerateContentConfig.
func FunctionDecls(c *genai.GenerateContentConfig) (ret []*genai.FunctionDeclaration) {
	if c == nil {
		return nil
	}
	for _, t := range c.Tools {
		ret = append(ret, t.FunctionDeclarations...)
	}
	return ret
}

func Must[T agent.Agent](a T, err error) T {
	if err != nil {
		panic(err)
	}
	return a
}

func AppendInstructions(r *model.LLMRequest, instructions ...string) {
	if len(instructions) == 0 {
		return
	}

	inst := strings.Join(instructions, "\n\n")

	if r.Config == nil {
		r.Config = &genai.GenerateContentConfig{}
	}

	if r.Config.SystemInstruction == nil {
		r.Config.SystemInstruction = genai.NewContentFromText(inst, genai.RoleUser)
	} else {
		r.Config.SystemInstruction.Parts = append(r.Config.SystemInstruction.Parts, genai.NewPartFromText(inst))
	}
}
