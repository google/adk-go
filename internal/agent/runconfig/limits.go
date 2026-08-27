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
	"fmt"
	"log"
	"os"
	"strconv"

	"google.golang.org/adk/v2/agent"
)

const (
	// DefaultMaxLLMCalls is the model-call budget an invocation gets when the
	// caller does not choose one. Matches adk-python's default.
	DefaultMaxLLMCalls = 500

	// MaxLLMCallsEnvVar overrides DefaultMaxLLMCalls. Matches adk-python.
	MaxLLMCallsEnvVar = "ADK_MAX_LLM_CALLS"
)

// ResolveMaxLLMCalls turns the value a caller set on a public run config into
// the limit to enforce.
//
// A positive value is used as given. A negative value means no limit and is
// passed through. Zero, which is what a caller gets from an empty struct
// literal, resolves to the environment override if it parses, and to
// DefaultMaxLLMCalls otherwise.
func ResolveMaxLLMCalls(v int) int {
	if v != 0 {
		return v
	}
	raw, ok := os.LookupEnv(MaxLLMCallsEnvVar)
	if !ok {
		return DefaultMaxLLMCalls
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		log.Printf("adk: invalid %s value %q, using the default %d", MaxLLMCallsEnvVar, raw, DefaultMaxLLMCalls)
		return DefaultMaxLLMCalls
	}
	return n
}

// IncrementAndEnforceLLMCallsLimit counts one model call against the
// invocation's budget and reports whether the budget is now exceeded.
//
// It counts first and then checks, so a limit of n permits exactly n calls. A
// nil receiver, which happens when an agent is run directly rather than through
// the runner, imposes no limit, and so does a non-positive limit.
func (c *RunConfig) IncrementAndEnforceLLMCallsLimit() error {
	if c == nil {
		return nil
	}
	calls := c.llmCalls.Add(1)
	if c.MaxLLMCalls <= 0 || calls <= int64(c.MaxLLMCalls) {
		return nil
	}
	return fmt.Errorf("%w: limit is %d", agent.ErrLLMCallsLimitExceeded, c.MaxLLMCalls)
}
