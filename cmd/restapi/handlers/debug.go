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

// DebugAPIController is the controller for the Debug API.
type DebugAPIController struct{}

// TraceDict returns the debug information for the session in form of dictionary.
func (*DebugAPIController) TraceDict(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

// EventGraph returns the debug information for the session and session events in form of graph.
func (*DebugAPIController) EventGraph(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}
