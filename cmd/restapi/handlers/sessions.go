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

	"google.golang.org/adk/sessionservice"
)

func unimplemented(rw http.ResponseWriter, req *http.Request) {
	rw.WriteHeader(http.StatusNotImplemented)
}

type SessionsApiController struct {
	service sessionservice.Service
}

func New(service sessionservice.Service) *SessionsApiController {
	return &SessionsApiController{service: service}
}

func (c *SessionsApiController) CreateSession(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

func (*SessionsApiController) DeleteSession(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

// GetSession handles receiving a sesion from the system.
func (*SessionsApiController) GetSession(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}

func (*SessionsApiController) ListSessions(rw http.ResponseWriter, req *http.Request) {
	unimplemented(rw, req)
}
