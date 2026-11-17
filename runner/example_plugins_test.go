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

package runner_test

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

// Plugin names are unique within a runner: New refuses a second plugin with
// a name already taken. Two plugins built by the same constructor need
// different names.
func ExamplePluginConfig_uniqueNames() {
	first, err := plugin.New(plugin.Config{Name: "audit"})
	if err != nil {
		log.Fatal(err)
	}
	second, err := plugin.New(plugin.Config{Name: "audit"})
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: fixedReplyModel{}})
	if err != nil {
		log.Fatal(err)
	}
	_, err = runner.New(runner.Config{
		AppName:        "app",
		Agent:          a,
		SessionService: session.InMemoryService(),
		PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{first, second}},
	})
	fmt.Println(err)

	// Output:
	// failed to create plugin manager: plugin with name 'audit' already registered
}

// Close calls every plugin's CloseFunc in registration order, waiting for
// each at most PluginConfig.CloseTimeout, and reports every plugin that
// failed in one error. A plugin that has not returned in time is reported
// with an error that matches context.DeadlineExceeded; its CloseFunc goes on
// running in the background.
func ExampleRunner_Close_plugins() {
	release := make(chan struct{})
	defer close(release)
	slow, err := plugin.New(plugin.Config{
		Name: "slow_exporter",
		CloseFunc: func() error {
			<-release
			return nil
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	broken, err := plugin.New(plugin.Config{
		Name:      "broken_exporter",
		CloseFunc: func() error { return errors.New("disk full") },
	})
	if err != nil {
		log.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: fixedReplyModel{}})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:        "app",
		Agent:          a,
		SessionService: session.InMemoryService(),
		PluginConfig: runner.PluginConfig{
			Plugins:      []*plugin.Plugin{slow, broken},
			CloseTimeout: 50 * time.Millisecond,
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	err = r.Close()
	fmt.Println(err)
	fmt.Println("timed out:", errors.Is(err, context.DeadlineExceeded))

	// Output:
	// Failed to close plugins: 'slow_exporter', 'broken_exporter'
	// error closing plugin 'slow_exporter': no result after 50ms: context deadline exceeded
	// error closing plugin 'broken_exporter': disk full
	// timed out: true
}
