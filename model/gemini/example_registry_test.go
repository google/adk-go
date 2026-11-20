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

package gemini_test

import (
	"context"
	"fmt"
	"log"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
)

// Importing package gemini registers its connector for the Gemini model
// names, so model.NewLLM resolves such a name. The model creates its genai
// client on first use, so resolving a name needs no credentials.
func ExampleIsGeminiConnector() {
	m, err := model.NewLLM(context.Background(), "gemini-2.5-flash")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(m.Name(), gemini.IsGeminiConnector(m))
	fmt.Println(gemini.IsGeminiConnector(nil))
	// Output:
	// gemini-2.5-flash true
	// false
}
