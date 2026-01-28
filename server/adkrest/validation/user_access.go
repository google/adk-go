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

package validation

import (
	"net/http"
)

// UserAccessValidator is an interface for user access validation
type UserAccessValidator interface {
	ValidateUserAccess(req *http.Request, appName, userID string) error
}

// UserAccessValidatorFunc is an adapter to allow a function to be used as a UserAccessValidator
type UserAccessValidatorFunc func(req *http.Request, appName, userID string) error

// ValidateUserAccess implements the UserAccessValidator interface for UserAccessValidatorFunc
func (f UserAccessValidatorFunc) ValidateUserAccess(req *http.Request, appName, userID string) error {
	return f(req, appName, userID)
}
