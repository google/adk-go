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

package sessionutils

import (
	"encoding/json"
	"fmt"

	"google.golang.org/adk/v2/tool/toolauth"
)

// CheckRequestedAuthConfigs returns an error when a request in configs would
// not read back from storage as a [toolauth.Config]. A struct literal is not
// checked when it is built, as adk-python checks an AuthConfig, and
// [toolauth.Config.MarshalJSON] accepts configs that
// [toolauth.Config.UnmarshalJSON] refuses. The in-memory, database and Vertex
// AI session services call this before they append an event, so none of them
// stores a request it could not load again, and all of them refuse the same
// events.
//
// The configs are read back with every number as a float64, as the Vertex AI
// service stores them through structpb, so an int64 such as an ExpiresAt that
// the conversion rounds out of range is refused too.
func CheckRequestedAuthConfigs(configs map[string]toolauth.Config) error {
	if len(configs) == 0 {
		return nil
	}
	b, err := json.Marshal(configs)
	if err != nil {
		return fmt.Errorf("encode requested auth configs: %w", err)
	}
	var stored any
	if err := json.Unmarshal(b, &stored); err != nil {
		return fmt.Errorf("decode requested auth configs: %w", err)
	}
	if b, err = json.Marshal(stored); err != nil {
		return fmt.Errorf("encode requested auth configs: %w", err)
	}
	var decoded map[string]toolauth.Config
	if err := json.Unmarshal(b, &decoded); err != nil {
		return fmt.Errorf("requested auth configs do not decode: %w", err)
	}
	return nil
}
