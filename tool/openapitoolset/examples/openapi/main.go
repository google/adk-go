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

// Package main demonstrates generating ADK tools from an OpenAPI document.
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"

	"google.golang.org/adk/tool/openapitoolset"
)

const document = `{
  "openapi": "3.0.3",
  "info": {"title": "Widget service", "version": "1.0.0"},
  "servers": [{"url": "/v1"}],
  "paths": {
    "/widgets/{id}": {
      "get": {
        "operationId": "getWidget",
        "summary": "Get a widget by ID",
        "parameters": [{
          "name": "id",
          "in": "path",
          "required": true,
          "schema": {"type": "string"}
        }],
        "responses": {"200": {"description": "Widget"}}
      }
    }
  }
}`

func main() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/openapi.json" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprint(w, document)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	toolset, err := openapitoolset.NewFromURL(context.Background(), server.URL+"/openapi.json", openapitoolset.Config{
		HTTPClient: server.Client(),
	})
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := toolset.Close(); err != nil {
			log.Printf("close toolset: %v", err)
		}
	}()

	tools, err := toolset.Tools(nil)
	if err != nil {
		log.Fatal(err)
	}
	for _, generatedTool := range tools {
		fmt.Printf("%s: %s\n", generatedTool.Name(), generatedTool.Description())
	}
}
