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

package vertexai

import (
	"testing"

	"google.golang.org/adk/v2/memory"
)

func TestVertexAIService_AddEventsToMemory_NotSupported(t *testing.T) {
	v := &vertexAIService{}
	err := v.AddEventsToMemory(t.Context(), &memory.AddEventsToMemoryRequest{AppName: "app", UserID: "user"})
	if err == nil {
		t.Error("AddEventsToMemory() error = nil, want non-nil: vertexAIService does not support AddEventsToMemory")
	}
}
