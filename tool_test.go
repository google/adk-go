// Copyright 2025 Google LLC
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

package adk_test

import (
	"context"

	"github.com/google/adk-go"
)

func ExampleNewTool() {
	type SumArgs struct {
		A int `json:"a"` // an integer to sum
		B int `json:"b"` // another integer to sum
	}
	type SumResult struct {
		Sum int `json:"sum"` // the sum of two integers
	}

	handler := func(ctx context.Context, input SumArgs) (SumResult, error) {
		return SumResult{Sum: input.A + input.B}, nil
	}
	tool := adk.NewTool("sum", "sums two integers", handler)
	_ = tool // use the tool
}
