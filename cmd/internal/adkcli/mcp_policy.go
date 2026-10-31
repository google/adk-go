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

package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"strings"

	"google.golang.org/adk/v2/internal/configurable"
)

// Keep the policy path out of logs while retaining the filesystem error cause.
type mcpPolicyReadError struct{ cause error }

func (e *mcpPolicyReadError) Error() string { return "read MCP policy failed" }
func (e *mcpPolicyReadError) Unwrap() error { return e.cause }

// loadMCPPolicy consumes an optional leading --mcp-policy flag. Remaining flags
// belong to the launcher, including flags for its default console mode.
func loadMCPPolicy(ctx context.Context, args []string) (context.Context, []string, error) {
	if len(args) == 0 {
		return ctx, args, nil
	}
	flag, path, hasValue := strings.Cut(args[0], "=")
	if flag != "--mcp-policy" && flag != "-mcp-policy" {
		return ctx, args, nil
	}
	rest := args[1:]
	if !hasValue {
		if len(rest) == 0 || strings.HasPrefix(rest[0], "-") {
			return nil, nil, fmt.Errorf("--mcp-policy requires a policy file path")
		}
		path, rest = rest[0], rest[1:]
	}
	if path == "" {
		return nil, nil, fmt.Errorf("--mcp-policy requires a non-empty policy file path")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, &mcpPolicyReadError{err}
	}
	ctx, err = configurable.WithMCPPolicy(ctx, bytes.NewReader(data))
	if err != nil {
		return nil, nil, fmt.Errorf("load MCP policy: %w", err)
	}
	return ctx, rest, nil
}
