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

package handlers

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"google.golang.org/adk/evaluation"
)

// EvalHandler encapsulates evaluation HTTP handlers.
type EvalHandler struct {
	storage evaluation.Storage
	runner  *evaluation.Runner
}

// NewEvalHandler creates a new evaluation handler.
func NewEvalHandler(storage evaluation.Storage, runner *evaluation.Runner) *EvalHandler {
	return &EvalHandler{
		storage: storage,
		runner:  runner,
	}
}

// ListEvalSets retrieves all eval sets for an app.
func (h *EvalHandler) ListEvalSets(w http.ResponseWriter, r *http.Request) {
	appName := mux.Vars(r)["app_name"]

	evalSets, err := h.storage.ListEvalSets(r.Context(), appName)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	EncodeJSONResponse(evalSets, http.StatusOK, w)
}

// CreateEvalSet creates a new evaluation set.
func (h *EvalHandler) CreateEvalSet(w http.ResponseWriter, r *http.Request) {
	appName := mux.Vars(r)["app_name"]

	var evalSet evaluation.EvalSet
	if err := json.NewDecoder(r.Body).Decode(&evalSet); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Generate ID if not provided
	if evalSet.ID == "" {
		evalSet.ID = uuid.New().String()
	}

	evalSet.CreatedAt = time.Now()

	if err := h.storage.SaveEvalSet(r.Context(), appName, &evalSet); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	EncodeJSONResponse(evalSet, http.StatusCreated, w)
}

// GetEvalSet retrieves a specific eval set.
func (h *EvalHandler) GetEvalSet(w http.ResponseWriter, r *http.Request) {
	appName := mux.Vars(r)["app_name"]
	evalSetName := mux.Vars(r)["eval_set_name"]

	evalSet, err := h.storage.GetEvalSet(r.Context(), appName, evalSetName)
	if err != nil {
		if err == evaluation.ErrNotFound {
			http.Error(w, "eval set not found", http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	EncodeJSONResponse(evalSet, http.StatusOK, w)
}

// RunEvalSet executes an evaluation set.
func (h *EvalHandler) RunEvalSet(w http.ResponseWriter, r *http.Request) {
	appName := mux.Vars(r)["app_name"]
	evalSetName := mux.Vars(r)["eval_set_name"]

	var config evaluation.EvalConfig
	if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Load eval set
	evalSet, err := h.storage.GetEvalSet(r.Context(), appName, evalSetName)
	if err != nil {
		if err == evaluation.ErrNotFound {
			http.Error(w, "eval set not found", http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Run evaluation
	if h.runner == nil {
		http.Error(w, "evaluation runner not configured", http.StatusServiceUnavailable)
		return
	}

	result, err := h.runner.RunEvalSet(r.Context(), evalSet, &config)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	EncodeJSONResponse(result, http.StatusOK, w)
}

// ListEvalResults retrieves evaluation results.
func (h *EvalHandler) ListEvalResults(w http.ResponseWriter, r *http.Request) {
	appName := mux.Vars(r)["app_name"]

	results, err := h.storage.ListEvalSetResults(r.Context(), appName)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	EncodeJSONResponse(results, http.StatusOK, w)
}

// GetEvalResult retrieves a specific evaluation result.
func (h *EvalHandler) GetEvalResult(w http.ResponseWriter, r *http.Request) {
	resultID := mux.Vars(r)["result_id"]

	result, err := h.storage.GetEvalSetResult(r.Context(), resultID)
	if err != nil {
		if err == evaluation.ErrNotFound {
			http.Error(w, "eval result not found", http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	EncodeJSONResponse(result, http.StatusOK, w)
}

// DeleteEvalSet deletes an evaluation set.
func (h *EvalHandler) DeleteEvalSet(w http.ResponseWriter, r *http.Request) {
	appName := mux.Vars(r)["app_name"]
	evalSetName := mux.Vars(r)["eval_set_name"]

	if err := h.storage.DeleteEvalSet(r.Context(), appName, evalSetName); err != nil {
		if err == evaluation.ErrNotFound {
			http.Error(w, "eval set not found", http.StatusNotFound)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}
