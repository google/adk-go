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

package transfertoagenttool_test

import (
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/tool/toolutils"
	"google.golang.org/adk/v2/tool/transfertoagenttool"
)

func ExampleNew() {
	// List the tool in the Tools of an LLM agent that has no transfer targets
	// of its own; the flow gives an agent with sub-agents its own
	// transfer_to_agent tool.
	transfer := transfertoagenttool.New(transfertoagenttool.Config{
		AgentNames:            []string{"billing_agent", "support_agent"},
		IncludeTransferReason: true,
	})

	decl := transfer.(toolutils.Tool).Declaration()
	params := decl.ParametersJsonSchema.(*jsonschema.Schema)
	_, offersReason := params.Properties["transfer_reason"]

	fmt.Println(decl.Name)
	fmt.Println("agent_name enum:", params.Properties["agent_name"].Enum)
	fmt.Println("offers transfer_reason:", offersReason)
	// Output:
	// transfer_to_agent
	// agent_name enum: [billing_agent support_agent]
	// offers transfer_reason: true
}
