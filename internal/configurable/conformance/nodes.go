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

package conformance

import (
	"fmt"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/configurable"
)

func uppercaseFormatter(ctx agent.InvocationContext, input string) (string, error) {
	fmt.Println("in uppercase formatter")
	fmt.Printf("input: %v\n", input)
	return strings.ToUpper(input), nil
}

func RegisterNodeFunctions() error {
	configurable.RegisterNodeFunction("conformance.uppercase_formatter", uppercaseFormatter)
	return nil
}
