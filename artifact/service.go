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

// Package artifact provides a service for managing artifacts.
//
// An artifact is a file identified by an application name, a user ID, a session ID,
// and a filename. The service provides basic storage operations for artifacts,
// such as Save, Load, Delete, and List. It also supports versioning of artifacts.
package artifact

import (
	"context"
	"fmt"
	"strings"

	"google.golang.org/genai"
)

// Service is the artifact storage service.
type Service interface {
	// Save saves an artifact to the artifact service storage.
	// The artifact is a file identified by the app name, user ID, session ID, and fileName.
	// After saving the artifact, a revision ID is returned to identify the artifact version.
	Save(ctx context.Context, req *SaveRequest) (*SaveResponse, error)
	// Load loads an artifact from the storage.
	// The artifact is a file identified by the appName, userID, sessionID and fileName.
	Load(ctx context.Context, req *LoadRequest) (*LoadResponse, error)
	// Delete deletes an artifact. Deleting a non-existing entry is not an error.
	Delete(ctx context.Context, req *DeleteRequest) error
	// List lists all the artifact filenames within a session.
	List(ctx context.Context, req *ListRequest) (*ListResponse, error)
	// Versions lists all versions of an artifact.
	Versions(ctx context.Context, req *VersionsRequest) (*VersionsResponse, error)
}

// requiredField is an internal type to use on validate operations
type requiredField struct {
	Name  string
	Value string
}

// SaveRequest is the parameter for [ArtifactService.Save].
type SaveRequest struct {
	AppName, UserID, SessionID, FileName string
	// Part is the artifact to store.
	Part *genai.Part

	// Below are optional fields.

	// If set, the artifact will be saved with this version.
	// If unset, a new version will be created.
	Version int64
}

// validateRequiredStrings checks a slice of fields in order.
// It returns the names of any fields with empty values, preserving the original order.
func validateRequiredStrings(fields []requiredField) []string {
	var missingFields []string
	for _, field := range fields {
		if field.Value == "" {
			missingFields = append(missingFields, field.Name)
		}
	}
	return missingFields
}

// Validate checks if the struct is valid or if it is missing fields.
func (req *SaveRequest) Validate() error {
	// Define the fields to check in the desired order
	fieldsToCheck := []requiredField{
		{Name: "AppName", Value: req.AppName},
		{Name: "UserID", Value: req.UserID},
		{Name: "SessionID", Value: req.SessionID},
		{Name: "FileName", Value: req.FileName},
	}

	// Use the helper function for all required string fields
	missingFields := validateRequiredStrings(fieldsToCheck)

	// Perform checks that don't fit the helper
	if req.Part == nil {
		missingFields = append(missingFields, "Part")
	}

	// If the slice has any items, it means fields were missing.
	if len(missingFields) > 0 {
		return fmt.Errorf("invalid save request: missing required fields: %s", strings.Join(missingFields, ", "))
	}

	if req.Part.Text == "" && req.Part.InlineData == nil {
		return fmt.Errorf("invalid save request: Part.InlineData or Part.Text has to be set")
	}
	return nil
}

// SaveResponse is the return type of [ArtifactService.Save].
type SaveResponse struct {
	Version int64
}

// LoadRequest is the parameter for [ArtifactService.Load].
type LoadRequest struct {
	AppName, UserID, SessionID, FileName string

	// Below are optional fields.
	Version int64
}

// Validate checks if the struct is valid or if it is missing fields.
func (req *LoadRequest) Validate() error {
	// Define the fields to check in the desired order
	fieldsToCheck := []requiredField{
		{Name: "AppName", Value: req.AppName},
		{Name: "UserID", Value: req.UserID},
		{Name: "SessionID", Value: req.SessionID},
		{Name: "FileName", Value: req.FileName},
	}

	// Use the helper function for all required string fields
	missingFields := validateRequiredStrings(fieldsToCheck)

	// If the slice has any items, it means fields were missing.
	if len(missingFields) > 0 {
		return fmt.Errorf("invalid load request: missing required fields: %s", strings.Join(missingFields, ", "))
	}
	return nil
}

// LoadResponse is the return type of [ArtifactService.Load].
type LoadResponse struct {
	// Part is the artifact stored.
	Part *genai.Part
}

// DeleteRequest is the parameter for [ArtifactService.Delete].
type DeleteRequest struct {
	AppName, UserID, SessionID, FileName string

	// Below are optional fields.
	Version int64
}

// Validate checks if the struct is valid or if it is missing fields.
func (req *DeleteRequest) Validate() error {
	// Define the fields to check in the desired order
	fieldsToCheck := []requiredField{
		{Name: "AppName", Value: req.AppName},
		{Name: "UserID", Value: req.UserID},
		{Name: "SessionID", Value: req.SessionID},
		{Name: "FileName", Value: req.FileName},
	}

	// Use the helper function for all required string fields
	missingFields := validateRequiredStrings(fieldsToCheck)

	// If the slice has any items, it means fields were missing.
	if len(missingFields) > 0 {
		return fmt.Errorf("invalid delete request: missing required fields: %s", strings.Join(missingFields, ", "))
	}
	return nil
}

// ListRequest is the parameter for [ArtifactService.List].
type ListRequest struct {
	AppName, UserID, SessionID string
}

// Validate checks if the struct is valid or if it is missing a field.
func (req *ListRequest) Validate() error {
	// Define the fields to check in the desired order
	fieldsToCheck := []requiredField{
		{Name: "AppName", Value: req.AppName},
		{Name: "UserID", Value: req.UserID},
		{Name: "SessionID", Value: req.SessionID},
	}

	// Use the helper function for all required string fields
	missingFields := validateRequiredStrings(fieldsToCheck)

	// If the slice has any items, it means fields were missing.
	if len(missingFields) > 0 {
		return fmt.Errorf("invalid list request: missing required fields: %s", strings.Join(missingFields, ", "))
	}
	return nil
}

// ListResponse is the return type of [ArtifactService.List].
type ListResponse struct {
	FileNames []string
}

// VersionsRequest is the parameter for [ArtifactService.Versions].
type VersionsRequest struct {
	AppName, UserID, SessionID, FileName string
}

// Validate checks if the struct is valid or if its missing field
func (req *VersionsRequest) Validate() error {
	// Define the fields to check in the desired order
	fieldsToCheck := []requiredField{
		{Name: "AppName", Value: req.AppName},
		{Name: "UserID", Value: req.UserID},
		{Name: "SessionID", Value: req.SessionID},
		{Name: "FileName", Value: req.FileName},
	}

	// Use the helper function for all required string fields
	missingFields := validateRequiredStrings(fieldsToCheck)

	// If the slice has any items, it means fields were missing.
	if len(missingFields) > 0 {
		return fmt.Errorf("invalid versions request: missing required fields: %s", strings.Join(missingFields, ", "))
	}
	return nil
}

// VersionsResponse is the parameter for [ArtifactService.Versions].
type VersionsResponse struct {
	Versions []int64
}
