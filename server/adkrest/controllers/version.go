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

package controllers

import (
	"net/http"
	"runtime"

	"google.golang.org/adk/v2/internal/version"
)

// VersionAPIController serves the version handshake the ADK web UI performs on
// every page load.
type VersionAPIController struct{}

// NewVersionAPIController creates a new VersionAPIController.
func NewVersionAPIController() *VersionAPIController {
	return &VersionAPIController{}
}

// VersionResponse is the body of the version endpoint. The field names match
// the ADK API contract shared with adk-python, so the same web UI can talk to
// either server.
type VersionResponse struct {
	// Version is the ADK release this server was built from.
	Version string `json:"version"`
	// Language identifies the ADK implementation. The field exists so a
	// non-Python server can identify itself; ADK Go always reports "go".
	Language string `json:"language"`
	// LanguageVersion is the toolchain version, for example "go1.26.7".
	LanguageVersion string `json:"language_version"`
}

// VersionHandler reports the ADK version and implementation language.
func (c *VersionAPIController) VersionHandler(rw http.ResponseWriter, _ *http.Request) {
	EncodeJSONResponse(VersionResponse{
		Version:         version.Version,
		Language:        "go",
		LanguageVersion: runtime.Version(),
	}, http.StatusOK, rw)
}
