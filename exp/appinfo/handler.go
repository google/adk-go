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

package appinfo

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/agent"
)

// experimentalWarning is logged the first time the endpoint serves a request.
// Its text is the notice adk-python's experimental decorator attaches to a
// feature (src/google/adk/utils/feature_decorator.py), so the two runtimes warn
// in the same words.
const experimentalWarning = "[EXPERIMENTAL] /apps/{app_name}/app-info: This feature is experimental and may change or be removed in future versions without notice. It may introduce breaking changes at any time."

// Handler returns the handler for GET /apps/{app_name}/app-info. It takes the
// app name from the gorilla/mux route variable app_name, loads that app's root
// agent from loader, and answers with its [AppInfo] as JSON.
//
// It answers 404 for an app loader does not serve and 503 when loader is nil.
//
// The first request it serves logs a warning that the endpoint is
// experimental, once per returned handler, as adk-python warns the first time
// an experimental feature is used. Mounted behind authentication, as the REST
// server mounts it, a request that is refused never reaches it and does not
// use up the warning.
func Handler(loader agent.Loader) http.HandlerFunc {
	var warnOnce sync.Once
	return func(w http.ResponseWriter, r *http.Request) {
		warnOnce.Do(func() { log.Print(experimentalWarning) })
		if loader == nil {
			http.Error(w, "no agent loader configured", http.StatusServiceUnavailable)
			return
		}
		appName := mux.Vars(r)["app_name"]
		root, err := loader.LoadAgent(appName)
		if err != nil || root == nil {
			http.Error(w, "app not found", http.StatusNotFound)
			return
		}

		// Encoded before anything is written, so a value JSON cannot represent
		// answers 500 rather than 200 with an empty body.
		body, err := json.Marshal(build(r.Context(), appName, root))
		if err != nil {
			// The type only: the error text can quote a tool's schema.
			log.Printf("app-info: encoding the description of an app failed: %T", err)
			http.Error(w, "encoding the app description failed", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json; charset=UTF-8")
		_, _ = w.Write(body)
	}
}
