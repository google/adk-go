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

package session_test

import (
	"encoding/json"
	"fmt"
	"log"

	"google.golang.org/adk/v2/session"
)

// A widget travels in EventActions.RenderUIWidgets, in render order. A nil
// Payload is written as an empty object, and an event without widgets has no
// "renderUiWidgets" key.
func ExampleUIWidget() {
	actions := session.EventActions{RenderUIWidgets: []session.UIWidget{
		{ID: "status", Provider: "mcp", Payload: map[string]any{"resource_uri": "ui://status"}},
		{ID: "spinner", Provider: "mcp"},
	}}
	b, err := json.Marshal(actions.RenderUIWidgets)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(b))

	b, err = json.Marshal(session.EventActions{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(b))
	// Output:
	// [{"id":"status","provider":"mcp","payload":{"resource_uri":"ui://status"}},{"id":"spinner","provider":"mcp","payload":{}}]
	// {}
}
