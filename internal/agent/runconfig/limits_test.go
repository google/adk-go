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

package runconfig

import (
	"errors"
	"sync"
	"testing"

	"google.golang.org/adk/v2/agent"
)

func TestResolveMaxLLMCalls(t *testing.T) {
	tests := []struct {
		name  string
		in    int
		env   string
		setEn bool
		want  int
	}{
		{name: "zero takes the default", in: 0, want: DefaultMaxLLMCalls},
		{name: "zero takes the environment override", in: 0, env: "7", setEn: true, want: 7},
		{name: "zero ignores an unparsable override", in: 0, env: "lots", setEn: true, want: DefaultMaxLLMCalls},
		{name: "zero honours a negative override", in: 0, env: "-1", setEn: true, want: -1},
		{name: "an explicit value wins over the environment", in: 5, env: "7", setEn: true, want: 5},
		{name: "a negative value is passed through", in: -1, want: -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.setEn {
				t.Setenv(MaxLLMCallsEnvVar, tt.env)
			}
			if got := ResolveMaxLLMCalls(tt.in); got != tt.want {
				t.Errorf("ResolveMaxLLMCalls(%d) = %d, want %d", tt.in, got, tt.want)
			}
		})
	}
}

func TestIncrementAndEnforceLLMCallsLimit(t *testing.T) {
	t.Run("a limit of n permits exactly n calls", func(t *testing.T) {
		c := &RunConfig{MaxLLMCalls: 2}
		for i := 1; i <= 2; i++ {
			if err := c.IncrementAndEnforceLLMCallsLimit(); err != nil {
				t.Fatalf("call %d returned %v, want nil", i, err)
			}
		}
		err := c.IncrementAndEnforceLLMCallsLimit()
		if !errors.Is(err, agent.ErrLLMCallsLimitExceeded) {
			t.Errorf("call 3 returned %v, want it to wrap ErrLLMCallsLimitExceeded", err)
		}
	})

	t.Run("non-positive limits do not bound", func(t *testing.T) {
		for _, limit := range []int{0, -1} {
			c := &RunConfig{MaxLLMCalls: limit}
			for i := 0; i < 100; i++ {
				if err := c.IncrementAndEnforceLLMCallsLimit(); err != nil {
					t.Fatalf("limit %d bounded the run at call %d: %v", limit, i, err)
				}
			}
		}
	})

	t.Run("a nil run config does not bound", func(t *testing.T) {
		var c *RunConfig
		if err := c.IncrementAndEnforceLLMCallsLimit(); err != nil {
			t.Errorf("nil receiver returned %v, want nil", err)
		}
	})

	t.Run("the counter is shared across concurrent agents", func(t *testing.T) {
		// Sub-agents in one invocation share the run config, so the budget is
		// for the invocation rather than for each agent.
		const limit = 50
		c := &RunConfig{MaxLLMCalls: limit}

		var wg sync.WaitGroup
		var mu sync.Mutex
		allowed := 0
		for i := 0; i < limit*2; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if err := c.IncrementAndEnforceLLMCallsLimit(); err == nil {
					mu.Lock()
					allowed++
					mu.Unlock()
				}
			}()
		}
		wg.Wait()

		if allowed != limit {
			t.Errorf("allowed %d calls, want %d", allowed, limit)
		}
	})
}
