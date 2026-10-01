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

package shared

import (
	"fmt"

	"google.golang.org/genai"
)

// EnsureFunctionToolOnly rejects a nil tool, one declaring no functions, and
// one carrying a non-function tool neither API is sent; idx names it.
func EnsureFunctionToolOnly(idx int, tool *genai.Tool) error {
	if tool == nil {
		return fmt.Errorf("openai: tool %d is nil", idx)
	}
	if tool.Retrieval != nil || tool.GoogleSearch != nil || tool.GoogleSearchRetrieval != nil ||
		tool.GoogleMaps != nil || tool.EnterpriseWebSearch != nil ||
		tool.URLContext != nil || tool.ComputerUse != nil || tool.CodeExecution != nil {
		return fmt.Errorf("openai: non-function tools are not supported (tool %d)", idx)
	}
	if len(tool.FunctionDeclarations) == 0 {
		return fmt.Errorf("openai: tool %d does not declare any functions", idx)
	}
	return nil
}
