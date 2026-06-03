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

package controllers

import (
	"net/http"
	"strconv"

	"github.com/gorilla/mux"

	"google.golang.org/adk/artifact"
	"google.golang.org/adk/server/adkrest/internal/models"
)

// ArtifactsAPIController is the controller for the Artifacts API.
type ArtifactsAPIController struct {
	artifactService artifact.Service
}

func NewArtifactsAPIController(artifactService artifact.Service) *ArtifactsAPIController {
	return &ArtifactsAPIController{artifactService: artifactService}
}

// ListArtifactsHandler lists all the artifact filenames within a session.
func (c *ArtifactsAPIController) ListArtifactsHandler(rw http.ResponseWriter, req *http.Request) {
	vars := mux.Vars(req)
	sessionID, err := models.SessionIDFromHTTPParameters(vars)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusBadRequest)
		return
	}
	if sessionID.ID == "" {
		http.Error(rw, "session_id parameter is required", http.StatusBadRequest)
		return
	}
	resp, err := c.artifactService.List(req.Context(), &artifact.ListRequest{
		AppName:   sessionID.AppName,
		UserID:    sessionID.UserID,
		SessionID: sessionID.ID,
	})
	if err != nil {
		http.Error(rw, err.Error(), http.StatusInternalServerError)
		return
	}
	files := resp.FileNames
	if files == nil {
		files = []string{}
	}
	EncodeJSONResponse(files, http.StatusOK, rw)
}

// LoadArtifactHandler gets an artifact from the artifact service storage.
func (c *ArtifactsAPIController) LoadArtifactHandler(rw http.ResponseWriter, req *http.Request) {
	vars := mux.Vars(req)
	sessionID, err := models.SessionIDFromHTTPParameters(vars)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusBadRequest)
		return
	}
	if sessionID.ID == "" {
		http.Error(rw, "session_id parameter is required", http.StatusBadRequest)
		return
	}
	artifactName := vars["artifact_name"]
	if artifactName == "" {
		http.Error(rw, "artifact_name parameter is required", http.StatusBadRequest)
		return
	}
	loadReq := &artifact.LoadRequest{
		AppName:   sessionID.AppName,
		UserID:    sessionID.UserID,
		SessionID: sessionID.ID,
		FileName:  artifactName,
	}

	queryParams := req.URL.Query()
	version := queryParams.Get("version")
	if version != "" {
		versionInt, err := strconv.Atoi(version)
		if err != nil {
			http.Error(rw, "version parameter must be an integer", http.StatusBadRequest)
			return
		}
		loadReq.Version = int64(versionInt)
	}

	resp, err := c.artifactService.Load(req.Context(), loadReq)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusInternalServerError)
		return
	}
	EncodeJSONResponse(resp.Part, http.StatusOK, rw)
}

// LoadArtifactVersionHandler gets an artifact from the artifact service storage with specified version.
func (c *ArtifactsAPIController) LoadArtifactVersionHandler(rw http.ResponseWriter, req *http.Request) {
	vars := mux.Vars(req)
	sessionID, err := models.SessionIDFromHTTPParameters(vars)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusBadRequest)
		return
	}
	if sessionID.ID == "" {
		http.Error(rw, "session_id parameter is required", http.StatusBadRequest)
		return
	}
	artifactName := vars["artifact_name"]
	if artifactName == "" {
		http.Error(rw, "artifact_name parameter is required", http.StatusBadRequest)
		return
	}
	version := vars["version"]

	if version == "" {
		http.Error(rw, "version parameter is required", http.StatusBadRequest)
		return
	}

	versionInt, err := strconv.Atoi(version)
	if err != nil {
		http.Error(rw, "version parameter must be an integer", http.StatusBadRequest)
		return
	}

	loadReq := &artifact.LoadRequest{
		AppName:   sessionID.AppName,
		UserID:    sessionID.UserID,
		SessionID: sessionID.ID,
		FileName:  artifactName,
		Version:   int64(versionInt),
	}

	resp, err := c.artifactService.Load(req.Context(), loadReq)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusInternalServerError)
		return
	}
	EncodeJSONResponse(resp.Part, http.StatusOK, rw)
}

// DeleteArtifactHandler handles deleting an artifact.
func (c *ArtifactsAPIController) DeleteArtifactHandler(rw http.ResponseWriter, req *http.Request) {
	vars := mux.Vars(req)
	sessionID, err := models.SessionIDFromHTTPParameters(vars)
	if err != nil {
		http.Error(rw, err.Error(), http.StatusBadRequest)
		return
	}
	if sessionID.ID == "" {
		http.Error(rw, "session_id parameter is required", http.StatusBadRequest)
		return
	}
	artifactName := vars["artifact_name"]
	if artifactName == "" {
		http.Error(rw, "artifact_name parameter is required", http.StatusBadRequest)
		return
	}
	err = c.artifactService.Delete(req.Context(), &artifact.DeleteRequest{
		AppName:   sessionID.AppName,
		UserID:    sessionID.UserID,
		SessionID: sessionID.ID,
		FileName:  artifactName,
	})
	if err != nil {
		http.Error(rw, err.Error(), http.StatusInternalServerError)
		return
	}
	EncodeJSONResponse(nil, http.StatusOK, rw)
}
