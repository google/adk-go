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

// package agentengine brings functionality of serving commands for AgentEngine-deployed code
package agentengine

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/gorilla/mux"

	"google.golang.org/protobuf/types/known/structpb"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/server/agentengine/controllers"
	"google.golang.org/adk/server/agentengine/controllers/method"
	"google.golang.org/adk/server/agentengine/internal/routers"
)

// NewHandler creates and returns an http.Handler for the AgentEngine API.
// Handles both streaming and non-streaming versions
func NewHandler(config *launcher.Config, sseWriteTimeout time.Duration, maxPayloadSize int64, agentEngineID string) (http.Handler, error) {
	router := mux.NewRouter().StrictSlash(true)

	nonStreamAgentEngineController, err := controllers.NewAgentEngineAPIController(config.SessionService, sseWriteTimeout, maxPayloadSize,
		listNonStreamHandlers(config, agentEngineID))
	if err != nil {
		return nil, fmt.Errorf("controllers.NewAgentEngineAPIController failed (for non-streaming): %v", err)
	}

	streamAgentEngineController, err := controllers.NewAgentEngineAPIController(config.SessionService, sseWriteTimeout, maxPayloadSize,
		listStreamHandlers(config, agentEngineID))
	if err != nil {
		return nil, fmt.Errorf("controllers.NewAgentEngineAPIController failed (for streaming): %v", err)
	}

	setupRouter(router,
		routers.NewReasoningEngineAPIRouter(nonStreamAgentEngineController),
		routers.NewStreamReasoningEngineAPIRouter(streamAgentEngineController),
	)

	methods, err := ListClassMethods()
	if err != nil {
		return nil, fmt.Errorf("ListClassMethods() failed: %v", err)
	}

	log.Println("Supported methods:")
	for _, m := range methods {
		sb := &strings.Builder{}
		err = json.NewEncoder(sb).Encode(m)
		if err != nil {
			return nil, fmt.Errorf("json.NewEncoder failed: %v", err)
		}
		log.Println(sb.String())
	}

	return router, nil
}

func setupRouter(router *mux.Router, subrouters ...routers.Router) *mux.Router {
	routers.SetupSubRouters(router, subrouters...)
	return router
}

// listNonStreamHandlers returnes a list of handlers for non-streaming methods
func listNonStreamHandlers(config *launcher.Config, agentEngineID string) []method.MethodHandler {
	return []method.MethodHandler{
		method.NewCreateSessionHandler(config.SessionService, agentEngineID, "async_create_session", "async"),
		method.NewGetSessionHandler(config.SessionService, agentEngineID, "async_get_session", "async"),
		method.NewListSessionHandler(config.SessionService, agentEngineID, "async_list_sessions", "async"),
		method.NewDeleteSessionHandler(config.SessionService, agentEngineID, "async_delete_session", "async"),
	}
}

// listStreamHandlers returnes a list of handlers for streaming methods
func listStreamHandlers(config *launcher.Config, agentEngineID string) []method.MethodHandler {
	return []method.MethodHandler{
		method.NewStreamQueryHandler(config, agentEngineID, "async_stream_query", "async_stream"),
	}
}

// ListClassMethods returns a list of structs, each describing a supported method. It is used to provide this information during the deployment
func ListClassMethods() ([]*structpb.Struct, error) {
	// create fake config with defaults, just to use list(Non)StreamHandlers
	config := &launcher.Config{}

	result := []*structpb.Struct{}

	handlers := slices.Concat(listNonStreamHandlers(config, ""), listStreamHandlers(config, ""))
	for _, handler := range handlers {
		m, err := handler.Metadata()
		if err != nil {
			return nil, err
		}
		result = append(result, m)
	}
	return result, nil
}
