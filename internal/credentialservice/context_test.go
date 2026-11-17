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

package credentialservice_test

import (
	"context"
	"testing"

	api "google.golang.org/adk/v2/credentialservice"
	"google.golang.org/adk/v2/internal/credentialservice"
	"google.golang.org/adk/v2/tool/toolauth"
)

// namedService is a service that tells instances apart by name.
type namedService struct{ name string }

func (*namedService) LoadCredential(context.Context, string, string, *toolauth.Config) (*toolauth.Credential, error) {
	return nil, nil
}

func (*namedService) SaveCredential(context.Context, string, string, *toolauth.Config) error {
	return nil
}

type otherKey struct{}

func TestWithServiceFromContext(t *testing.T) {
	a := &namedService{name: "a"}
	b := &namedService{name: "b"}
	withA := credentialservice.WithService(t.Context(), a)
	canceledChild, cancel := context.WithCancel(withA)
	cancel()
	var nilCtx context.Context

	tests := []struct {
		name string
		ctx  context.Context
		want api.Service
	}{
		{name: "no service attached", ctx: t.Context(), want: nil},
		{name: "service attached", ctx: withA, want: a},
		{name: "child with a value", ctx: context.WithValue(withA, otherKey{}, "v"), want: a},
		{name: "canceled child", ctx: canceledChild, want: a},
		{name: "later service replaces earlier", ctx: credentialservice.WithService(withA, b), want: b},
		{name: "nil service replaces earlier", ctx: credentialservice.WithService(withA, nil), want: nil},
		{name: "nil context", ctx: nilCtx, want: nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := credentialservice.FromContext(tc.ctx); got != tc.want {
				t.Errorf("FromContext() = %v, want %v", got, tc.want)
			}
		})
	}
}
