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

package toolconfirmation_test

import (
	"encoding/json"
	"fmt"
	"log"

	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// A confirmation without a payload has no "payload" key in its JSON form, as
// in adk-python.
func ExampleToolConfirmation() {
	for _, c := range []toolconfirmation.ToolConfirmation{
		{Hint: "Delete the file?"},
		{Hint: "Transfer funds?", Payload: map[string]any{"amount": 100}},
	} {
		b, err := json.Marshal(c)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(string(b))
	}
	// Output:
	// {"hint":"Delete the file?","confirmed":false}
	// {"hint":"Transfer funds?","confirmed":false,"payload":{"amount":100}}
}
