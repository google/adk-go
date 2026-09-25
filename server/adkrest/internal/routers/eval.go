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

package routers

import (
	"net/http"

	"google.golang.org/adk/v2/server/adkrest/controllers"
)

// EvalAPIRouter defines the routes for the Eval API.
//
// ADK Go has no evaluation implementation. The routes exist so the endpoints
// the web UI calls are recognised and answered deliberately, with 501 and a
// readable body, rather than falling through to a bare 404 that looks like a
// wrong path prefix.
type EvalAPIRouter struct{}

// evalDetail must contain controllers.NotInstalledDetail. The web UI checks a
// failed eval run's error detail for that phrase and shows its own
// eval-unavailable notice when it matches.
const evalDetail = "agent evaluation is " + controllers.NotInstalledDetail +
	" in ADK Go. Use adk-python for eval workflows."

// Routes returns the routes for the Eval API.
func (r *EvalAPIRouter) Routes() Routes {
	unavailable := controllers.NewNotImplementedHandler("eval", evalDetail)

	routes := Routes{
		Route{
			Name:        "GetEvalMetricsInfo",
			Methods:     []string{http.MethodGet},
			Pattern:     DevPrefix + "/metrics-info",
			HandlerFunc: controllers.MetricsInfoHandler,
		},
	}

	// The UI spells this resource both ways: it creates an eval set at
	// "eval-sets" but reads, updates and deletes at "eval_sets". Both have to
	// route, exactly as they do upstream.
	evalRoutes := []struct {
		name    string
		methods []string
		pattern string
	}{
		{"ListEvalSets", []string{http.MethodGet}, "/eval_sets"},
		{"CreateEvalSet", []string{http.MethodPost}, "/eval-sets"},
		{"CreateEvalSetUnderscore", []string{http.MethodPost}, "/eval_sets"},
		{"GetEvalSet", []string{http.MethodGet}, "/eval_sets/{eval_set_id}"},
		{"DeleteEvalSet", []string{http.MethodDelete}, "/eval_sets/{eval_set_id}"},
		{"ListEvalCases", []string{http.MethodGet}, "/eval_sets/{eval_set_id}/evals"},
		{"GetEvalCase", []string{http.MethodGet}, "/eval_sets/{eval_set_id}/evals/{eval_case_id}"},
		{"UpdateEvalCase", []string{http.MethodPut}, "/eval_sets/{eval_set_id}/evals/{eval_case_id}"},
		{"DeleteEvalCase", []string{http.MethodDelete}, "/eval_sets/{eval_set_id}/evals/{eval_case_id}"},
		{"AddSessionToEvalSet", []string{http.MethodPost}, "/eval_sets/{eval_set_id}/add_session"},
		{"RunEval", []string{http.MethodPost}, "/eval_sets/{eval_set_id}/run_eval"},
		{"ListEvalResults", []string{http.MethodGet}, "/eval_results"},
		{"GetEvalResult", []string{http.MethodGet}, "/eval_results/{eval_result_id}"},
	}
	for _, e := range evalRoutes {
		routes = append(routes, Route{
			Name:        e.name,
			Methods:     e.methods,
			Pattern:     DevPrefix + e.pattern,
			HandlerFunc: unavailable,
		})
	}

	// Pre-v2 clients addressed the eval endpoints without the /dev prefix.
	// These keep those three paths routed and answering 501, as they always
	// did; only the body changed, from empty to the JSON every eval endpoint
	// now returns.
	legacyRoutes := []struct {
		name    string
		methods []string
		pattern string
	}{
		{"ListEvalSetsLegacyPath", []string{http.MethodGet}, "/apps/{app_name}/eval_sets"},
		{"ListEvalResultsLegacyPath", []string{http.MethodGet}, "/apps/{app_name}/eval_results"},
		{"CreateEvalSetLegacyPath", []string{http.MethodPost}, "/apps/{app_name}/eval_sets/{eval_set_name}"},
	}
	for _, l := range legacyRoutes {
		routes = append(routes, Route{
			Name:        l.name,
			Methods:     l.methods,
			Pattern:     l.pattern,
			HandlerFunc: unavailable,
		})
	}

	return routes
}
