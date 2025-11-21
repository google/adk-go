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

/*
Package a2asrv provides a configurable A2A protocol server implementation.

The default implementation can be created using NewRequestHandler. The function takes a single required
[AgentExecutor] dependency and a variable number of [RequestHandlerOption]-s used to customize handler behavior.

AgentExecutor implementation is responsible for invoking the agent, translating its outputs
to a2a core types and writing them to the provided [eventqueue.Queue]. A2A server will be reading
data from the queue, processing it and notifying connected clients.

RequestHandler is transport-agnostic and needs to be wrapped in a transport-specific translation layer
like [github.com/a2aproject/a2a-go/a2agrpc.Handler]. JSONRPC transport implementation can be created using NewJSONRPCHandler function
and registered with the standard [http.Server]:

	handler := a2asrv.NewRequestHandler(
		agentExecutor,
		a2asrv.WithTaskStore(customDB),
		a2asrv.WithPushNotifications(configStore, sender),
		a2asrv.WithCallInterceptor(customMiddleware),
		...
	)

	mux := http.NewServeMux()
	mux.Handle("/invoke", a2asrv.NewJSONRPCHandler(handler))

The package provides utilities for serving public [a2a.AgentCard]-s. These return handler implementations
which can be registered with a standard http server. Since the card is public, CORS policy allows requests from any domain.

	mux.Handle(a2asrv, a2asrv.NewStaticAgentCardHandler(card))

	// or for more advanced use cases

	mux.Handle(a2asrv, a2asrv.NewAgentCardHandler(producer))
*/
package a2asrv
