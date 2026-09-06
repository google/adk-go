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

import "net/http"

// This file answers the developer-only endpoints that the ADK web UI calls but
// ADK Go does not implement: evaluation, the agent builder, and the tests tab.
//
// They are registered rather than left unrouted on purpose. An unrouted path
// returns a bare 404 that a client cannot distinguish from talking to the wrong
// server or the wrong path prefix, and it hides the fact that the UI expects
// the endpoint at all. Registering them makes the gap explicit, gives each
// response a body a developer can read, and makes the behavior testable.

// ErrorResponse is a machine-readable error body.
type ErrorResponse struct {
	Error string `json:"error"`
	// Detail explains what a caller can do about it.
	Detail string `json:"detail,omitempty"`
}

// NotInstalledDetail is the marker the ADK web UI looks for to tell "this
// server cannot run evals" apart from "this eval run failed".
//
// The UI inspects the error detail of a failed eval run for the substring "not
// installed" and, when it matches, raises its own eval-unavailable notice
// rather than reporting a transport error. Any detail the eval endpoints return
// must therefore contain this phrase.
//
// An earlier version of this file answered the eval endpoints with 404 instead,
// on the theory that the UI hides its Evals tab on a 404. It does not: the eval
// tab component emits that decision on a shouldShowTab output, and the side
// panel that hosts the component subscribes to four of its outputs but not that
// one. The signal goes nowhere, so 501 is used, consistently with every other
// unimplemented developer endpoint.
const NotInstalledDetail = "not installed"

// NewNotImplementedHandler returns a handler that reports 501 for a route ADK
// Go serves but does not implement. 501 tells a client the endpoint is
// recognised and the feature is missing, which is the difference between "this
// server is a different version" and "this server cannot do that".
func NewNotImplementedHandler(feature, detail string) http.HandlerFunc {
	return func(rw http.ResponseWriter, _ *http.Request) {
		EncodeJSONResponse(ErrorResponse{
			Error:  feature + " is not implemented in ADK Go",
			Detail: detail,
		}, http.StatusNotImplemented, rw)
	}
}

// MetricsInfoResponse is the body of the eval metrics-info endpoint.
type MetricsInfoResponse struct {
	MetricsInfo []any `json:"metricsInfo"`
}

// MetricsInfoHandler reports the eval metrics this server supports, which is
// none. An empty list is the accurate answer and keeps the client from logging
// a transport error for a question that has a valid empty result.
func MetricsInfoHandler(rw http.ResponseWriter, _ *http.Request) {
	EncodeJSONResponse(MetricsInfoResponse{MetricsInfo: []any{}}, http.StatusOK, rw)
}

// AgentBuilderConfigHandler answers the agent builder's request for an app's
// stored configuration.
//
// It returns an empty document, not an error. ADK Go agents are defined in Go
// code, so there is no editable configuration to hand back, and the UI reads an
// empty body as "nothing to edit here" and disables the builder toggle. An
// error status would make it log a failure for a question that has a correct
// empty answer.
func AgentBuilderConfigHandler(rw http.ResponseWriter, _ *http.Request) {
	rw.Header().Set("Content-Type", "text/plain; charset=UTF-8")
	rw.WriteHeader(http.StatusOK)
}
