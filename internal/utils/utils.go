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
	"encoding/json"
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

// UnwrapResponse extracts the original value from a FunctionResponse payload.
// A sole single-key wrapper — {"result": v} (adk-python and the web frontend),
// {"response": v} or {"payload": v} (adk-go) — is unwrapped, with string values
// JSON-parsed when possible; anything else, including a multi-key map such as
// a tool confirmation's {"confirmed": b, "payload": v}, passes through whole.
// Mirrors adk-python _unwrap_response, extended with the adk-go keys for
// cross-runtime sessions.
//
// The workflow engine's three resume paths — the inbound turn in runner and in
// workflowagent, and the replay from session history — all decode through here,
// so a reply read back from history yields the same value as the same reply
// taken from the inbound turn. Tool confirmation is decoded separately, by
// llminternal's confirmation request processor, which produces the Confirmed
// flag the tool layer enforces; that decision must not be routed through here,
// since this function has no notion of it.
func UnwrapResponse(data map[string]any) any {
	if data == nil {
		return nil // untyped nil, not a nil map: callers compare against nil
	}
	if len(data) != 1 {
		return data
	}
	for _, key := range []string{"result", "response", "payload"} {
		v, ok := data[key]
		if !ok {
			continue
		}
		if s, isStr := v.(string); isStr {
			var parsed any
			if err := json.Unmarshal([]byte(s), &parsed); err == nil {
				return parsed
			}
			return s
		}
		return v
	}
	return data
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
