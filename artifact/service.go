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
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"

	"google.golang.org/genai"
)

// Service is the artifact storage service.
type Service interface {
	// Save saves an artifact to the artifact service storage.
	// The artifact is a file identified by the app name, user ID, session ID, and fileName.
	// After saving the artifact, a revision ID is returned to identify the artifact version.
	// Implementations that assign versions optimistically may fail to claim one
	// while other writers save the same artifact; that failure is transient and
	// the call can be retried.
	Save(ctx context.Context, req *SaveRequest) (*SaveResponse, error)
	// Load loads an artifact from the storage.
	// The artifact is a file identified by the appName, userID, sessionID and fileName.
	Load(ctx context.Context, req *LoadRequest) (*LoadResponse, error)
	// Delete deletes an artifact. If req.Version is 0, all versions are deleted,
	// and deleting an artifact that does not exist is not an error. If req.Version
	// is set and that version does not exist, implementations may return an error.
	Delete(ctx context.Context, req *DeleteRequest) error
	// List lists the artifact filenames visible to a session: the session's own
	// files and the user-scoped ("user:" prefixed) files of the user. When
	// req.SessionID is empty, List returns only the user-scoped filenames.
	List(ctx context.Context, req *ListRequest) (*ListResponse, error)
	// Versions lists all versions of an artifact.
	Versions(ctx context.Context, req *VersionsRequest) (*VersionsResponse, error)
	// GetArtifactVersion gets the metadata for a specific version of an artifact.
	GetArtifactVersion(ctx context.Context, req *GetArtifactVersionRequest) (*GetArtifactVersionResponse, error)
}

// ErrInputValidation classifies an artifact request or an artifact reference
// that carries invalid input: a missing or malformed app name, user ID,
// session ID or file name, an unsupported part, or an artifact:// reference
// that is malformed, leaves the caller's scope or is followed too deep. The
// Validate methods of the request types, [ValidatePathSegment] and the
// in-memory service wrap it, so callers test for it with [errors.Is]. The
// messages never echo the caller's values.
var ErrInputValidation = errors.New("invalid input")

// validationError carries a message and wraps ErrInputValidation without
// changing the message: Error returns msg byte for byte.
type validationError struct {
	msg string
	err error
}

func (e *validationError) Error() string { return e.msg }
func (e *validationError) Unwrap() error { return e.err }

// invalidInput returns an error with the text msg that wraps
// ErrInputValidation.
func invalidInput(msg string) error {
	return &validationError{msg: msg, err: ErrInputValidation}
}

// requiredField is an internal type to use on validate operations
type requiredField struct {
	Name  string
	Value string
}

// SaveRequest is the parameter for [Service.Save].
type SaveRequest struct {
	AppName, UserID, SessionID, FileName string
	// Part is the artifact to store.
	Part *genai.Part

	// Below are optional fields.

	// CustomMetadata contains caller-defined metadata persisted with the artifact.
	// Service implementations must convert each value to its string representation
	// when saving it and return the value as a string.
	CustomMetadata map[string]any

	// Deprecated: Save ignores Version. Every Save creates a new version and
	// returns it in SaveResponse.Version.
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

// artifactName holds the fields of a request that name one artifact.
type artifactName struct {
	appName, userID, sessionID, fileName string
}

// missingFields returns the names of the empty required fields, in the order
// the missing-field error lists them. A "user:" file name names a user-scoped
// artifact, so the session ID is not required for it.
func (n artifactName) missingFields() []string {
	fields := []requiredField{
		{Name: "AppName", Value: n.appName},
		{Name: "UserID", Value: n.userID},
	}
	if !fileHasUserNamespace(n.fileName) {
		fields = append(fields, requiredField{Name: "SessionID", Value: n.sessionID})
	}
	fields = append(fields, requiredField{Name: "FileName", Value: n.fileName})
	return validateRequiredStrings(fields)
}

// validatePaths path-checks the app name, the user ID and, unless the file
// name is "user:" prefixed, the session ID, in that order.
func (n artifactName) validatePaths() error {
	if err := validateAppAndUser(n.appName, n.userID); err != nil {
		return err
	}
	if fileHasUserNamespace(n.fileName) {
		return nil
	}
	return ValidatePathSegment(n.sessionID, "session_id")
}

// validateAppAndUser path-checks the app name, then the user ID.
func validateAppAndUser(appName, userID string) error {
	if err := ValidatePathSegment(appName, "app_name"); err != nil {
		return err
	}
	return ValidatePathSegment(userID, "user_id")
}

// Validate checks the request: AppName, UserID, FileName and Part are required,
// and SessionID is required unless FileName has the "user:" prefix. AppName and
// UserID, and SessionID when it is required, are checked with
// [ValidatePathSegment] in that order. A "user:" file name makes the session
// irrelevant: an empty SessionID is accepted and a non-empty one is not
// checked. The part must carry text or inline data, or have no field set:
// genai.Part cannot tell empty text from unset text, so the services in this
// module store a part with no field set as empty text. A part with any other
// field set and neither text nor inline data, such as file data alone, is
// refused. Every error wraps [ErrInputValidation].
func (req *SaveRequest) Validate() error {
	return req.validate(false)
}

// ValidateAllowingFileData checks the request with the rules of
// [SaveRequest.Validate], except that the part may also be file data: an
// external file URI or an artifact:// reference. It is for a [Service] that
// stores file data itself, such as the Cloud Storage service; Validate stays
// the check for services that do not.
//
// File data whose FileURI is empty is refused, as adk-python's Cloud Storage
// service does, unless the part also carries inline data or non-empty text,
// which the service stores instead of the file data. The method checks
// the form of the part only: the service still checks an artifact:// URI with
// [ParseAndValidateReferenceURI]. Every error wraps [ErrInputValidation] and
// leaves the caller's values out of its message. It is safe for concurrent use
// as long as the request is not modified meanwhile.
func (req *SaveRequest) ValidateAllowingFileData() error {
	if err := req.validate(true); err != nil {
		return err
	}
	part := req.Part
	if part.InlineData == nil && part.Text == "" && part.FileData != nil && part.FileData.FileURI == "" {
		return invalidInput("invalid save request: Part.FileData.FileURI has to be set (Artifact file_data must have a file_uri.)")
	}
	return nil
}

// validate is the shared request validation, with allowRef tolerating a file
// data part, an artifact reference or external data. Validate() (allowRef
// false) accepts a part with text, a part with inline data, or a part with no
// field set, which the services store as empty text, and rejects every other
// part: the shared contract of services that do not store file data. The
// services that store it set allowRef: the in-memory service directly, and
// others through ValidateAllowingFileData.
func (req *SaveRequest) validate(allowRef bool) error {
	name := artifactName{appName: req.AppName, userID: req.UserID, sessionID: req.SessionID, fileName: req.FileName}
	missingFields := name.missingFields()

	// Perform checks that don't fit the helper
	if req.Part == nil {
		missingFields = append(missingFields, "Part")
	}

	// If the slice has any items, it means fields were missing.
	if len(missingFields) > 0 {
		return invalidInput(fmt.Sprintf("invalid save request: missing required fields: %s", strings.Join(missingFields, ", ")))
	}

	if err := name.validatePaths(); err != nil {
		return err
	}

	// File data, an artifact reference or external data, carries neither
	// Part.Text nor Part.InlineData; the content rule tolerates it only for
	// the services that store file data and stays strict for every other
	// caller. A part whose every field is zero is empty text: genai.Part cannot
	// tell Text "" from an unset Text, and adk-python accepts Part(text="").
	if req.Part.Text == "" && req.Part.InlineData == nil && !(allowRef && req.Part.FileData != nil) && !reflect.ValueOf(*req.Part).IsZero() {
		return invalidInput("invalid save request: Part.InlineData or Part.Text has to be set")
	}

	// Validate that FileName doesn't contain path separators
	if err := ValidateFileName(req.FileName); err != nil {
		return err
	}
	return nil
}

// ValidateFileName reports whether name is usable as an artifact file name:
// it returns an error wrapping [ErrInputValidation] when name contains a
// path separator ("/" or "\") and nil otherwise. The error text does not
// contain name.
func ValidateFileName(name string) error {
	if strings.Contains(name, "/") || strings.Contains(name, "\\") {
		return invalidInput("invalid name: filename cannot contain path separators")
	}
	return nil
}

// SaveResponse is the return type of [Service.Save].
type SaveResponse struct {
	Version int64
}

// LoadRequest is the parameter for [Service.Load].
type LoadRequest struct {
	AppName, UserID, SessionID, FileName string

	// Below are optional fields.
	Version int64
}

// Validate has the same field rules as [SaveRequest.Validate], without the part.
func (req *LoadRequest) Validate() error {
	name := artifactName{appName: req.AppName, userID: req.UserID, sessionID: req.SessionID, fileName: req.FileName}
	// If the slice has any items, it means fields were missing.
	if missingFields := name.missingFields(); len(missingFields) > 0 {
		return invalidInput(fmt.Sprintf("invalid load request: missing required fields: %s", strings.Join(missingFields, ", ")))
	}

	if err := name.validatePaths(); err != nil {
		return err
	}

	// Validate that FileName doesn't contain path separators
	if err := ValidateFileName(req.FileName); err != nil {
		return err
	}

	return nil
}

// LoadResponse is the return type of [Service.Load].
type LoadResponse struct {
	// Part is the artifact stored.
	Part *genai.Part
}

// DeleteRequest is the parameter for [Service.Delete].
type DeleteRequest struct {
	AppName, UserID, SessionID, FileName string

	// Below are optional fields.
	Version int64
}

// Validate has the same field rules as [SaveRequest.Validate], without the part.
func (req *DeleteRequest) Validate() error {
	name := artifactName{appName: req.AppName, userID: req.UserID, sessionID: req.SessionID, fileName: req.FileName}
	// If the slice has any items, it means fields were missing.
	if missingFields := name.missingFields(); len(missingFields) > 0 {
		return invalidInput(fmt.Sprintf("invalid delete request: missing required fields: %s", strings.Join(missingFields, ", ")))
	}

	if err := name.validatePaths(); err != nil {
		return err
	}

	// Validate that FileName doesn't contain path separators
	if err := ValidateFileName(req.FileName); err != nil {
		return err
	}

	return nil
}

// ListRequest is the parameter for [Service.List]. AppName and UserID are
// required and checked with [ValidatePathSegment]. SessionID is optional; an
// empty SessionID asks for the user-scoped file names only.
type ListRequest struct {
	AppName, UserID, SessionID string
}

// Validate checks the request. AppName and UserID are required and path-checked
// in that order. SessionID is path-checked only when it is not empty; an empty
// SessionID is valid when AppName and UserID are set, and the error lists
// SessionID among the missing fields only when AppName or UserID is missing
// too. Every error wraps [ErrInputValidation].
func (req *ListRequest) Validate() error {
	// Define the fields to check in the desired order
	fieldsToCheck := []requiredField{
		{Name: "AppName", Value: req.AppName},
		{Name: "UserID", Value: req.UserID},
	}

	// Use the helper function for all required string fields
	missingFields := validateRequiredStrings(fieldsToCheck)

	// If the slice has any items, it means fields were missing.
	if len(missingFields) > 0 {
		if req.SessionID == "" {
			missingFields = append(missingFields, "SessionID")
		}
		return invalidInput(fmt.Sprintf("invalid list request: missing required fields: %s", strings.Join(missingFields, ", ")))
	}
	if err := validateAppAndUser(req.AppName, req.UserID); err != nil {
		return err
	}
	if req.SessionID == "" {
		return nil
	}
	return ValidatePathSegment(req.SessionID, "session_id")
}

// ListResponse is the return type of [Service.List].
type ListResponse struct {
	FileNames []string
}

// VersionsRequest is the parameter for [Service.Versions].
type VersionsRequest struct {
	AppName, UserID, SessionID, FileName string
}

// Validate has the same field rules as [SaveRequest.Validate], without the part.
func (req *VersionsRequest) Validate() error {
	name := artifactName{appName: req.AppName, userID: req.UserID, sessionID: req.SessionID, fileName: req.FileName}
	// If the slice has any items, it means fields were missing.
	if missingFields := name.missingFields(); len(missingFields) > 0 {
		return invalidInput(fmt.Sprintf("invalid versions request: missing required fields: %s", strings.Join(missingFields, ", ")))
	}

	if err := name.validatePaths(); err != nil {
		return err
	}

	// Validate that FileName doesn't contain path separators
	if err := ValidateFileName(req.FileName); err != nil {
		return err
	}

	return nil
}

// VersionsResponse is the return type of [Service.Versions].
type VersionsResponse struct {
	Versions []int64
}

// ArtifactVersion contains metadata describing a specific version of an artifact.
// Service implementations must populate its fields as documented below.
type ArtifactVersion struct {
	Version int64

	// CanonicalURI is a stable identity for the stored payload. Implementations
	// with a native scheme should use it, such as gs:// for Google Cloud Storage;
	// other implementations should define a stable service-specific scheme. It
	// is not necessarily a download endpoint or an authenticated HTTP URL.
	CanonicalURI string

	// Implementations must return a non-nil map containing only string values.
	// It is empty when no custom metadata was saved.
	CustomMetadata map[string]any

	// CreateTime is when this artifact version was created.
	CreateTime time.Time

	// MimeType is the media type associated with the stored part. It may be empty
	// when the part was saved without a media type, or when the stored part is a
	// reference to another artifact, whose media type is only known once the
	// reference is loaded.
	MimeType string
}

// GetArtifactVersionRequest is the parameter for [Service.GetArtifactVersion].
type GetArtifactVersionRequest struct {
	AppName, UserID, SessionID, FileName string

	// Below are optional fields.
	Version int64
}

// Validate has the same field rules as [SaveRequest.Validate], without the part.
func (req *GetArtifactVersionRequest) Validate() error {
	name := artifactName{appName: req.AppName, userID: req.UserID, sessionID: req.SessionID, fileName: req.FileName}
	// If the slice has any items, it means fields were missing.
	if missingFields := name.missingFields(); len(missingFields) > 0 {
		return invalidInput(fmt.Sprintf("invalid get artifact version request: missing required fields: %s", strings.Join(missingFields, ", ")))
	}

	if err := name.validatePaths(); err != nil {
		return err
	}

	// Validate that FileName doesn't contain path separators
	if err := ValidateFileName(req.FileName); err != nil {
		return err
	}

	return nil
}

// GetArtifactVersionResponse is the return type of [Service.GetArtifactVersion].
type GetArtifactVersionResponse struct {
	ArtifactVersion *ArtifactVersion
}
