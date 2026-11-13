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

package session

import "encoding/json"

// UIWidget is a request from a tool or callback to the UI host to render a
// widget alongside an event, recorded in [EventActions.RenderUIWidgets].
//
// ID identifies the widget within one event's actions. Provider names the
// system that renders it, such as "mcp". Payload holds provider-specific
// data; a nil Payload means an empty one and is written to JSON as {}.
// The zero value is a widget with no ID, no provider and an empty payload.
// A UIWidget is not safe for concurrent use.
type UIWidget struct {
	ID       string         `json:"id"`
	Provider string         `json:"provider"`
	Payload  map[string]any `json:"payload"`
}

// MarshalJSON implements [json.Marshaler]. It writes a nil Payload as {}, as
// adk-python writes a widget whose payload was never set.
func (w UIWidget) MarshalJSON() ([]byte, error) {
	// alias strips this method, so json does not recurse into it.
	type alias UIWidget
	if w.Payload == nil {
		w.Payload = map[string]any{}
	}
	return json.Marshal(alias(w))
}

// UnmarshalJSON implements [json.Unmarshaler]. Unknown keys are ignored, as
// for adk-go's other session types, and a missing or null payload decodes as
// an empty map.
func (w *UIWidget) UnmarshalJSON(data []byte) error {
	type alias UIWidget
	if err := json.Unmarshal(data, (*alias)(w)); err != nil {
		return err
	}
	if w.Payload == nil {
		w.Payload = map[string]any{}
	}
	return nil
}
