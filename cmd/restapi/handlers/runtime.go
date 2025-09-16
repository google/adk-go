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

package handlers

import (
	"net/http"
)

// RuntimeAPIController is the controller for the Runtime API.
type RuntimeAPIController struct{}

// RunAgent executes a non-streaming agent run for a given session and message.
func (*RuntimeAPIController) RunAgent(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

// RunAgentSSE executes an agent run and streams the resulting events using Server-Sent Events (SSE).
func (*RuntimeAPIController) RunAgentSSE(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}
