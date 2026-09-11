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

// Package authz provides authorization for the ADK REST API.
package authz

import (
	"context"
	"errors"
	"net/http"
)

// ErrUnauthorized should be returned by [Authorizer] CanActAsUser to indicate
// a failure in authorization.
var ErrUnauthorized = errors.New("authz: unauthorized")

// Authorizer provides information whether the [Caller] can act as a userID
// It is used in ADK REST API (adkrest), which has the userID in paths.
type Authorizer interface {
	// CanActAsUser returns nil iff the [Caller]] from the ctx can act as a userID. Otherwise returns an error
	// WARNING: avoid returning PII data in the error, it will be send back to the client.
	CanActAsUser(ctx context.Context, userID string) error
}

// WriteHTTPStatusForAuthError writes a 403 Forbidden response, appending the
// error message when one is provided. Forbidden, not Unauthorized (401): the
// caller is authenticated but not permitted to act as the requested user.
func WriteHTTPStatusForAuthError(rw http.ResponseWriter, err error) {
	errMsg := "forbidden"
	if err != nil {
		errMsg = errMsg + ": " + err.Error()
	}
	http.Error(rw, errMsg, http.StatusForbidden)
}
