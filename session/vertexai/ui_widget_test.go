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

package vertexai

import (
	"testing"

	"google.golang.org/adk/v2/session"
)

// The aiplatform EventActions message has no field for UI widgets, so an
// event whose only payload is its widgets must be written with raw_event or
// the widgets are lost on reload.
func TestEventNeedsRawEvent_RenderUIWidgets(t *testing.T) {
	event := &session.Event{Actions: session.EventActions{
		RenderUIWidgets: []session.UIWidget{{ID: "w1", Provider: "mcp"}},
	}}
	if !eventNeedsRawEvent(event) {
		t.Error("eventNeedsRawEvent() = false for an event with UI widgets, so the widgets would be lost on reload")
	}
}
