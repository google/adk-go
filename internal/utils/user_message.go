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

package utils

import (
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

// FindUserMessageForInvocation returns the content of the user message that
// started the invocation invocationID: the first live event (see
// [ApplyRewinds]) of that invocation authored "user" whose content has at
// least one part and no part carrying a function response. A message with
// no text part, such as an image-only turn, qualifies. An empty invocationID
// matches the events of every invocation. It returns nil when no event
// qualifies.
//
// The returned content is the stored event's; callers must not modify it.
// It ports adk-python's Runner._find_user_message_for_invocation
// (runners.py).
func FindUserMessageForInvocation(events session.Events, invocationID string) *genai.Content {
	for _, ev := range ApplyRewinds(slices.Collect(events.All())) {
		if ev.Author != "user" || ev.Content == nil || len(ev.Content.Parts) == 0 {
			continue
		}
		if invocationID != "" && ev.InvocationID != invocationID {
			continue
		}
		if len(FunctionResponses(ev.Content)) > 0 {
			continue
		}
		return ev.Content
	}
	return nil
}
