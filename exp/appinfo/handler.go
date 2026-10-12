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
	"log"
	"net/http"
	"sync"

	"github.com/gorilla/mux"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/server/adkrest/controllers"
)

// experimentalWarning is logged the first time the endpoint serves a request.
const experimentalWarning = "[EXPERIMENTAL] /apps/{app_name}/app-info: This feature is experimental and may change or be removed in future versions without notice. It may introduce breaking changes at any time."

// Handler returns the handler for GET /apps/{app_name}/app-info. It takes the
// app name from the gorilla/mux route variable app_name, loads that app's root
// agent from loader, and answers with its [AppInfo] as JSON.
//
// It answers 404 for an app loader does not serve and 503 when loader is nil.
//
// The first request it serves logs a warning that the endpoint is
// experimental, once per returned handler. Mounted behind authentication, as
// [NewLauncher] mounts it, a request that is refused never reaches it and does
// not use up the warning.
//
// The handler does no authentication or Origin and Host checking of its own.
// A server that mounts it directly is responsible for both.
func Handler(loader agent.Loader) http.HandlerFunc {
	var warnOnce sync.Once
	return func(w http.ResponseWriter, r *http.Request) {
		warnOnce.Do(func() { log.Print(experimentalWarning) }) //nolint:forbidigo // a notice to the operator, not an error, so there is nothing to return
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
		controllers.EncodeJSONResponse(build(r.Context(), appName, root), http.StatusOK, w)
	}
}
