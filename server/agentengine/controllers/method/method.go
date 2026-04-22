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

// package method defines MethodHandler which is used to serve requests for
// an application deployed on agent engine
package method

import (
	"context"
	"net/http"

	"google.golang.org/protobuf/types/known/structpb"
)

// MethodHandler is an interface which provides a structured way to serve methods on agentEngine
type MethodHandler interface {
	Name() string                                                             // Name returns a name of the method, by which you call it
	Handle(ctx context.Context, rw http.ResponseWriter, payload []byte) error // Handle provides a response to given payload
	Metadata() (*structpb.Struct, error)                                      // Metadata returns a struct which is used to specify application capabilities during the deployment
}
