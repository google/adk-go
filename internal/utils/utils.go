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
	"context"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
)

// TODO: split in proper files/packages.

const afFunctionCallIDPrefix = "adk-"

// PopulateClientFunctionCallID sets the function call ID field if it is empty.
// Since the ID field is optional, some models don't fill the field, but
// the LLMAgent depends on the IDs to map FunctionCall and FunctionResponse events
// in the event stream.
func PopulateClientFunctionCallID(ctx context.Context, c *genai.Content) {
	for _, fn := range FunctionCalls(c) {
		if fn.ID == "" {
			fn.ID = GenerateFunctionCallID(ctx)
		}
	}
}

// GenerateFunctionCallID generates a new function call ID. The ID is obtained
// through the platform package, so a UUID provider installed on ctx (see
// platform.WithUUIDProvider) controls it.
func GenerateFunctionCallID(ctx context.Context) string {
	return afFunctionCallIDPrefix + platform.NewUUID(ctx)
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

// AppendInstructions appends instructions to the [genai.GenerateContentConfig.SystemInstruction] system instruction.
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
		return
	}
	if len(r.Config.SystemInstruction.Parts) > 0 && r.Config.SystemInstruction.Parts[len(r.Config.SystemInstruction.Parts)-1].Text != "" {
		r.Config.SystemInstruction.Parts[len(r.Config.SystemInstruction.Parts)-1].Text += "\n\n" + inst
		return
	}
	r.Config.SystemInstruction.Parts = append(r.Config.SystemInstruction.Parts, genai.NewPartFromText(inst))
}

// IsProsePart reports whether p is plain text meant to be read, and nothing
// else.
//
// Exactly one field of a [genai.Part] is meant to be set, so a part carrying
// any of the actionable payloads is not prose whatever else is on it. Callers
// that filter on this drop such a part rather than reducing it to its text:
// the text is not what makes it dangerous, and dropping is the conservative
// half of the choice.
//
// A thought is not prose either. It is the model's private reasoning rather
// than anything it chose to say, and it should not be stored or replayed as
// though the model had said it.
func IsProsePart(p *genai.Part) bool {
	if p == nil || p.Text == "" || p.Thought {
		return false
	}
	return p.FunctionCall == nil &&
		p.FunctionResponse == nil &&
		p.ExecutableCode == nil &&
		p.CodeExecutionResult == nil &&
		p.FileData == nil &&
		p.InlineData == nil &&
		p.ToolCall == nil &&
		p.ToolResponse == nil
}

// EventBelongsToBranch reports whether an event on eventBranch is visible to an
// invocation running on invocationBranch.
//
// An event belongs to its own branch and to every descendant of it, so a child
// agent sees what its parent said and not the other way round. Branch nodes are
// delimited with a dot, and the prefix match requires that dot so that
// "agent_0" does not match "agent_00".
//
// The single definition, because prompt assembly and anything reasoning about
// what a prompt contains have to agree on it.
func EventBelongsToBranch(invocationBranch, eventBranch string) bool {
	if invocationBranch == "" || eventBranch == "" {
		return true
	}
	if eventBranch == invocationBranch {
		return true
	}
	return strings.HasPrefix(invocationBranch, eventBranch+".")
}
