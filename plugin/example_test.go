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

package plugin_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// greeter returns an agent that answers every message with "hello".
func greeter() agent.Agent {
	a, err := agent.New(agent.Config{
		Name: "greeter",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "greeter"
				ev.Content = genai.NewContentFromText("hello", genai.RoleModel)
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	return a
}

// errExportFailed is the failure of the exporter in ExampleConfig_failableAfterRun.
var errExportFailed = errors.New("export failed")

// A plugin whose end-of-run work can fail sets FailableAfterRunCallback
// instead of AfterRunCallback. Its error is a run error: the caller receives
// it after the run's last event, wrapped with the plugin's name, and every
// plugin's OnRunErrorCallback receives the same value.
func ExampleConfig_failableAfterRun() {
	exporter, err := plugin.New(plugin.Config{
		Name: "exporter",
		FailableAfterRunCallback: func(agent.InvocationContext) error {
			return errExportFailed
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	alerts, err := plugin.New(plugin.Config{
		Name: "alerts",
		OnRunErrorCallback: func(_ agent.InvocationContext, err error) {
			fmt.Println("alert: run failed:", errors.Is(err, errExportFailed))
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             greeter(),
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*plugin.Plugin{exporter, alerts}},
	})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("hi", genai.RoleUser)
	for ev, err := range r.Run(context.Background(), "user", "session", msg, agent.RunConfig{}) {
		if err != nil {
			fmt.Println("caller received error:", err)
			continue
		}
		fmt.Printf("caller received event: %s: %s\n", ev.Author, ev.Content.Parts[0].Text)
	}

	// Output:
	// caller received event: greeter: hello
	// caller received error: Error in plugin 'exporter' during 'after_run_callback' callback: export failed
	// alert: run failed: true
}

// errNotAllowed is the refusal of the gate in ExampleConfig_callbackError.
var errNotAllowed = errors.New("user is not allowed")

// An error that a plugin's BeforeRunCallback returns ends the run before any
// agent runs. The caller receives it wrapped with the plugin's name and the
// callback's adk-python name; errors.Is still finds the plugin's own error.
func ExampleConfig_callbackError() {
	gate, err := plugin.New(plugin.Config{
		Name: "gate",
		BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
			return nil, errNotAllowed
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "app",
		Agent:             greeter(),
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		PluginConfig:      runner.PluginConfig{Plugins: []*plugin.Plugin{gate}},
	})
	if err != nil {
		log.Fatal(err)
	}
	msg := genai.NewContentFromText("hi", genai.RoleUser)
	for _, err := range r.Run(context.Background(), "user", "session", msg, agent.RunConfig{}) {
		if err != nil {
			fmt.Println(err)
			fmt.Println("is errNotAllowed:", errors.Is(err, errNotAllowed))
			break
		}
	}

	// Output:
	// Error in plugin 'gate' during 'before_run_callback' callback: user is not allowed
	// is errNotAllowed: true
}

// New refuses a Config that sets both after-run forms.
func ExampleNew_bothAfterRunCallbacks() {
	_, err := plugin.New(plugin.Config{
		Name:                     "cleanup",
		AfterRunCallback:         func(agent.InvocationContext) {},
		FailableAfterRunCallback: func(agent.InvocationContext) error { return nil },
	})
	fmt.Println(err)

	// Output:
	// plugin: AfterRunCallback and FailableAfterRunCallback are both set
}
