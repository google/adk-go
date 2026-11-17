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

package toolcallintegrity_test

import (
	"fmt"
	"iter"
	"log"
	"maps"
	"slices"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/toolcallintegrity"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// A runner with the integrity plugin registered first. During key rotation
// the new key signs and the old key still verifies the calls it stamped.
func ExampleNew() {
	p, err := toolcallintegrity.New(toolcallintegrity.Config{
		SecretKey:        []byte("new-secret-key"),
		VerificationKeys: [][]byte{[]byte("old-secret-key")},
	})
	if err != nil {
		log.Fatal(err)
	}
	root, err := agent.New(agent.Config{
		Name: "assistant",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	_, err = runner.New(runner.Config{
		AppName:        "bank",
		Agent:          root,
		SessionService: session.InMemoryService(),
		PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(p.Name())

	// Output:
	// tool_call_integrity
}

// Copying an event into another session: the copy loses the source
// session's stamps and is marked restored, so the plugin skips it before a
// run and never runs its function calls.
func ExamplePrepareRestoredEvent() {
	ev := &session.Event{Author: "assistant"}
	ev.Content = &genai.Content{Role: "model", Parts: []*genai.Part{{
		FunctionCall: &genai.FunctionCall{ID: "fc-1", Name: "transfer_money", Args: map[string]any{"amount": 10}},
	}}}
	ev.CustomMetadata = map[string]any{
		"user_tag":               "imported",
		"__adk_internal_fc_hmac": map[string]any{"fc-1": "v1:..."},
	}

	restored := toolcallintegrity.PrepareRestoredEvent(ev)
	fmt.Println(slices.Sorted(maps.Keys(restored.CustomMetadata)))

	// Output:
	// [__adk_internal_restored_event user_tag]
}
