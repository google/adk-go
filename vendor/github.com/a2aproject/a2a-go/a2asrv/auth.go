// Copyright 2025 The A2A Authors
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

package a2asrv

// User can be attached to [CallContext] by authentication middleware.
type User interface {
	// Name returns a username.
	Name() string
	// Authenticated returns true if the request was authenticated.
	Authenticated() bool
}

// AuthenticatedUser is a simple implementation of [User] interface configurable with a username.
type AuthenticatedUser struct {
	UserName string
}

func (u *AuthenticatedUser) Name() string {
	return u.UserName
}

func (u *AuthenticatedUser) Authenticated() bool {
	return true
}

type unauthenticatedUser struct{}

func (unauthenticatedUser) Name() string {
	return ""
}

func (unauthenticatedUser) Authenticated() bool {
	return false
}
