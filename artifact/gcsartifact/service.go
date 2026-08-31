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

// Package gcsartifact provides a Google Cloud Storage (GCS) [artifact.Service].
//
// This package allows storing and retrieving artifacts in a GCS bucket.
// Artifacts are organized by application name, user ID, session ID, and filename,
// with support for versioning.
package gcsartifact

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"math/rand/v2"
	"net/http"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"cloud.google.com/go/storage"
	"golang.org/x/sync/errgroup"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"google.golang.org/adk/v2/artifact"
)

// gcsService is a google cloud storage implementation of the Service.
type gcsService struct {
	bucketName    string
	storageClient gcsClient
	bucket        gcsBucket
	// sleep waits for d or until ctx is done; a field so tests can stub out the
	// Save retry backoff. Never nil: every construction site sets it.
	sleep func(ctx context.Context, d time.Duration) error
}

// NewService creates a Google Cloud Storage service for the specified bucket.
func NewService(ctx context.Context, bucketName string, opts ...option.ClientOption) (artifact.Service, error) {
	storageClient, err := storage.NewClient(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create gcs service: %w", err)
	}
	// Wrap the real client
	clientWrapper := &gcsClientWrapper{client: storageClient}
	s := &gcsService{
		bucketName:    bucketName,
		storageClient: clientWrapper,
		bucket:        clientWrapper.bucket(bucketName),
		sleep:         sleepContext,
	}
	return s, nil
}

// fileHasUserNamespace checks if a filename indicates a user-namespaced blob.
func fileHasUserNamespace(filename string) bool {
	return strings.HasPrefix(filename, "user:")
}

// buildBlobName constructs the blob name in GCS.
func buildBlobName(appName, userID, sessionID, fileName string, version int64) string {
	if fileHasUserNamespace(fileName) {
		return fmt.Sprintf("%s/%s/user/%s/%d", appName, userID, fileName, version)
	}
	return fmt.Sprintf("%s/%s/%s/%s/%d", appName, userID, sessionID, fileName, version)
}

func buildBlobNamePrefix(appName, userID, sessionID, fileName string) string {
	if fileHasUserNamespace(fileName) {
		return fmt.Sprintf("%s/%s/user/%s/", appName, userID, fileName)
	}
	return fmt.Sprintf("%s/%s/%s/%s/", appName, userID, sessionID, fileName)
}

func buildSessionPrefix(appName, userID, sessionID string) string {
	return fmt.Sprintf("%s/%s/%s/", appName, userID, sessionID)
}

func buildUserPrefix(appName, userID string) string {
	return fmt.Sprintf("%s/%s/user/", appName, userID)
}

const (
	// maxSaveAttempts caps Save's retries; each retry means a concurrent writer
	// took the version we picked, so this is how many colliding writers we
	// tolerate before returning [ErrVersionConflict].
	maxSaveAttempts = 16
	// saveRetryBaseDelay and saveRetryMaxDelay bound the jittered backoff Save
	// waits between those retries.
	saveRetryBaseDelay = 10 * time.Millisecond
	saveRetryMaxDelay  = 1 * time.Second
)

// ErrVersionConflict is returned by Save when it cannot claim a new version
// within its retry budget because concurrent writers keep taking the version it
// picks. It is always wrapped, so test for it with [errors.Is]; a caller seeing
// it can safely retry the save.
var ErrVersionConflict = errors.New("artifact version conflict")

// Save implements [artifact.Service].
//
// GCS can't list-then-write atomically, so versions are assigned optimistically:
// Save writes version max+1 with a does-not-exist precondition and, if another
// writer already took that version, backs off and retries with a fresh one (up
// to [maxSaveAttempts]). It returns [ErrVersionConflict] if that budget is
// exhausted rather than overwriting an existing version. Each attempt costs one
// listing plus one upload.
//
// The scheme only holds if every writer uses the precondition: a bucket also
// written by a client that assigns versions unconditionally can still lose one.
func (s *gcsService) Save(ctx context.Context, req *artifact.SaveRequest) (*artifact.SaveResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName

	var lastErr error
	for attempt := range maxSaveAttempts {
		response, err := s.versions(ctx, &artifact.VersionsRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to list artifact versions: %w", err)
		}
		nextVersion := int64(1)
		if len(response.Versions) > 0 {
			nextVersion = slices.Max(response.Versions) + 1
		}

		blobName := buildBlobName(appName, userID, sessionID, fileName, nextVersion)
		obj := s.bucket.object(blobName).ifNotExist()
		err = writeArtifact(ctx, obj, req.Part)
		if err == nil {
			return &artifact.SaveResponse{Version: nextVersion}, nil
		}
		if !isPreconditionFailed(err) {
			return nil, fmt.Errorf("failed to save artifact: %w", err)
		}
		lastErr = err
		// Lost the race for this version. Back off (skip after the final attempt)
		// and retry with a fresh version.
		if attempt < maxSaveAttempts-1 {
			if err := s.sleep(ctx, backoffDelay(attempt)); err != nil {
				return nil, fmt.Errorf("failed to save artifact %q: %w", fileName, err)
			}
		}
	}
	return nil, fmt.Errorf("failed to save artifact %q after %d attempts (last error: %v): %w", fileName, maxSaveAttempts, lastErr, ErrVersionConflict)
}

// backoffDelay returns a full-jitter backoff for the given zero-based retry
// attempt, capped at saveRetryMaxDelay.
func backoffDelay(attempt int) time.Duration {
	d := saveRetryMaxDelay
	if attempt < 63 {
		if scaled := saveRetryBaseDelay << attempt; scaled > 0 && scaled < d {
			d = scaled
		}
	}
	return time.Duration(rand.Int64N(int64(d) + 1))
}

// sleepContext waits for d or until ctx is done, returning ctx.Err() if ctx is
// done first.
func sleepContext(ctx context.Context, d time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if d <= 0 {
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// writeArtifact streams part to obj. A precondition on obj surfaces as an error
// from Close, not Write.
func writeArtifact(ctx context.Context, obj gcsObject, part *genai.Part) (err error) {
	writer := obj.newWriter(ctx)
	defer func() {
		if closeErr := writer.Close(); closeErr != nil && err == nil {
			err = fmt.Errorf("failed to close blob writer: %w", closeErr)
		}
	}()

	if part.InlineData != nil {
		writer.SetContentType(part.InlineData.MIMEType)
		if _, err := writer.Write(part.InlineData.Data); err != nil {
			return fmt.Errorf("failed to write blob to GCS: %w", err)
		}
	} else {
		writer.SetContentType("text/plain")
		if _, err := writer.Write([]byte(part.Text)); err != nil {
			return fmt.Errorf("failed to write text to GCS: %w", err)
		}
	}
	return nil
}

// isPreconditionFailed reports whether err is a GCS precondition failure: HTTP
// 412 (*googleapi.Error) on the JSON API, or codes.FailedPrecondition on the
// gRPC transport. Both status.FromError and errors.As unwrap wrapped errors.
func isPreconditionFailed(err error) bool {
	var apiErr *googleapi.Error
	if errors.As(err, &apiErr) {
		return apiErr.Code == http.StatusPreconditionFailed
	}
	return status.Code(err) == codes.FailedPrecondition
}

// Delete implements [artifact.Service]
func (s *gcsService) Delete(ctx context.Context, req *artifact.DeleteRequest) error {
	err := req.Validate()
	if err != nil {
		return fmt.Errorf("request validation failed: %w", err)
	}
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	version := req.Version

	// Delete specific version
	if version != 0 {
		blobName := buildBlobName(appName, userID, sessionID, fileName, version)
		if err := s.bucket.object(blobName).delete(ctx); err != nil {
			return fmt.Errorf("failed to delete artifact: %w", err)
		}
		return nil
	}

	// Delete all versions
	response, err := s.versions(ctx, &artifact.VersionsRequest{
		AppName: req.AppName, UserID: req.UserID, SessionID: req.SessionID, FileName: req.FileName,
	})
	if err != nil {
		return fmt.Errorf("failed to fetch versions on delete artifact: %w", err)
	}

	g, gctx := errgroup.WithContext(ctx)

	// delete versions in parallel
	for _, version := range response.Versions {
		v := version // capture loop variable for goroutine

		g.Go(func() error {
			blobName := buildBlobName(appName, userID, sessionID, fileName, v)
			obj := s.bucket.object(blobName)
			if err := obj.delete(gctx); err != nil {
				return fmt.Errorf("failed to delete artifact %s: %w", blobName, err)
			}
			return nil // nil error indicates success for this goroutine
		})
	}

	return g.Wait()
}

// Load implements [artifact.Service]
func (s *gcsService) Load(ctx context.Context, req *artifact.LoadRequest) (_ *artifact.LoadResponse, err error) {
	err = req.Validate()
	if err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	version, err := s.resolveVersion(ctx, appName, userID, sessionID, fileName, req.Version)
	if err != nil {
		return nil, err
	}

	blobName := buildBlobName(appName, userID, sessionID, fileName, version)
	blob := s.bucket.object(blobName)

	// Check if the blob exists before trying to read it
	attrs, err := blob.attrs(ctx)
	if err != nil {
		if errors.Is(err, storage.ErrObjectNotExist) {
			return nil, fmt.Errorf("artifact '%s' not found: %w", blobName, fs.ErrNotExist)
		}
		return nil, fmt.Errorf("could not get blob attributes: %w", err)
	}

	// Create a reader to stream the blob's content
	reader, err := blob.newReader(ctx)
	if err != nil {
		return nil, fmt.Errorf("could not create reader for blob '%s': %w", blobName, err)
	}
	defer func() {
		if closeErr := reader.Close(); closeErr != nil && err == nil {
			err = fmt.Errorf("failed to close blob reader: %w", closeErr)
		}
	}()

	// Read all the content into a byte slice
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("could not read data from blob '%s': %w", blobName, err)
	}

	// Create the genai.Part and return the response.
	part := genai.NewPartFromBytes(data, attrs.ContentType)

	return &artifact.LoadResponse{Part: part}, nil
}

// fetchFilenamesFromPrefix is a reusable helper function.
func (s *gcsService) fetchFilenamesFromPrefix(ctx context.Context, prefix string, filenamesSet map[string]bool) error {
	// Add a guard clause to prevent a panic if a nil map is passed.
	if filenamesSet == nil {
		return fmt.Errorf("filenamesSet cannot be nil")
	}

	query := &storage.Query{
		Prefix: prefix,
	}
	// Only fill the attribute Name of the blob, the other attributes will have defaults.
	err := query.SetAttrSelection([]string{"Name"})
	if err != nil {
		return fmt.Errorf("error setting query attribute selection: %w", err)
	}
	blobsIterator := s.bucket.objects(ctx, query)

	for {
		blob, err := blobsIterator.next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return fmt.Errorf("error iterating blobs: %w", err)
		}
		segments := strings.Split(blob.Name, "/")
		if len(segments) < 2 {
			return fmt.Errorf("error iterating blobs: incorrect number of segments in path %q", blob.Name)
		}
		// Extract filename from path: appName/userId/sessionId/filename/version or appName/userId/user/filename/version
		// Note: filenames with path separators are rejected during validation (see service.go Validate methods)
		filename := segments[len(segments)-2]
		filenamesSet[filename] = true
	}

	return nil
}

// List implements [artifact.Service]
func (s *gcsService) List(ctx context.Context, req *artifact.ListRequest) (*artifact.ListResponse, error) {
	err := req.Validate()
	if err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	appName, userID, sessionID := req.AppName, req.UserID, req.SessionID
	filenamesSet := map[string]bool{}

	// Fetch filenames for the session.
	err = s.fetchFilenamesFromPrefix(ctx, buildSessionPrefix(appName, userID, sessionID), filenamesSet)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch session filenames: %w", err)
	}

	// Fetch filenames for the user.
	err = s.fetchFilenamesFromPrefix(ctx, buildUserPrefix(appName, userID), filenamesSet)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch user filenames: %w", err)
	}

	filenames := slices.Collect(maps.Keys(filenamesSet))
	sort.Strings(filenames)
	return &artifact.ListResponse{FileNames: filenames}, nil
}

// versions internal function that does not return error if versions are empty
func (s *gcsService) versions(ctx context.Context, req *artifact.VersionsRequest) (*artifact.VersionsResponse, error) {
	err := req.Validate()
	if err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName

	prefix := buildBlobNamePrefix(appName, userID, sessionID, fileName)
	query := &storage.Query{
		Prefix: prefix,
	}
	blobsIterator := s.bucket.objects(ctx, query)

	versions := make([]int64, 0)
	for {
		blob, err := blobsIterator.next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error iterating blobs: %w", err)
		}
		segments := strings.Split(blob.Name, "/")
		if len(segments) < 1 {
			return nil, fmt.Errorf("error iterating blobs: incorrect number of segments in path %q", blob.Name)
		}
		version, err := strconv.ParseInt(segments[len(segments)-1], 10, 64)
		// if the file version is not convertible to number, just ignore it
		if err != nil {
			continue
		}
		versions = append(versions, version)
	}
	return &artifact.VersionsResponse{Versions: versions}, nil
}

// Versions implements [artifact.Service] and returns an error if no versions are found.
func (s *gcsService) Versions(ctx context.Context, req *artifact.VersionsRequest) (*artifact.VersionsResponse, error) {
	response, err := s.versions(ctx, req)
	if err != nil {
		return nil, err
	}
	if len(response.Versions) == 0 {
		return nil, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
	}
	return response, nil
}

// resolveVersion returns the provided version if non-zero, otherwise finds the latest version.
func (s *gcsService) resolveVersion(ctx context.Context, appName, userID, sessionID, fileName string, version int64) (int64, error) {
	if version == 0 {
		response, err := s.versions(ctx, &artifact.VersionsRequest{
			AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName,
		})
		if err != nil {
			return 0, fmt.Errorf("failed to list artifact versions: %w", err)
		}
		if len(response.Versions) == 0 {
			return 0, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
		}
		version = slices.Max(response.Versions)
	}
	return version, nil
}

// GetArtifactVersion implements [artifact.Service] and returns the metadata for a specific version.
func (s *gcsService) GetArtifactVersion(ctx context.Context, req *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	err := req.Validate()
	if err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	version, err := s.resolveVersion(ctx, appName, userID, sessionID, fileName, req.Version)
	if err != nil {
		return nil, err
	}

	blobName := buildBlobName(appName, userID, sessionID, fileName, version)
	blob := s.bucket.object(blobName)

	attrs, err := blob.attrs(ctx)
	if err != nil {
		if errors.Is(err, storage.ErrObjectNotExist) {
			return nil, fmt.Errorf("artifact '%s' not found: %w", blobName, fs.ErrNotExist)
		}
		return nil, fmt.Errorf("could not get blob attributes: %w", err)
	}

	var canonicalURI string
	if attrs.MediaLink != "" {
		canonicalURI = attrs.MediaLink
	} else {
		canonicalURI = fmt.Sprintf("gs://%s/%s", s.bucketName, blobName)
	}

	customMeta := make(map[string]any)
	if attrs.Metadata != nil {
		for k, v := range attrs.Metadata {
			customMeta[k] = v
		}
	}

	return &artifact.GetArtifactVersionResponse{
		ArtifactVersion: &artifact.ArtifactVersion{
			Version:        version,
			CanonicalURI:   canonicalURI,
			CustomMetadata: customMeta,
			CreateTime:     attrs.Created,
			MimeType:       attrs.ContentType,
		},
	}, nil
}
