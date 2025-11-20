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

// Package gcs provides a Google Cloud Storage (GCS) implementation of the
// [artifact.Service] interface.
//
// This package allows storing and retrieving artifacts in a GCS bucket.
// Artifacts are organized by application name, user ID, session ID, and filename,
// with support for versioning.
package gcs

import (
	"context"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"slices"
	"sort"
	"strconv"
	"strings"

	"cloud.google.com/go/storage"
	"golang.org/x/sync/errgroup"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
	"google.golang.org/genai"

	"google.golang.org/adk/artifact"
)

// gcsService is a google cloud storage implementation of the Service.
type gcsService struct {
	bucketName    string
	storageClient gcsClient
	bucket        gcsBucket
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

// Save implements [artifact.Service]
func (s *gcsService) Save(ctx context.Context, req *artifact.SaveRequest) (_ *artifact.SaveResponse, err error) {
	err = req.Validate()
	if err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	newArtifact := req.Part

	nextVersion := int64(1)

	// TODO race condition, could use mutex but it's a remote resource so the issue would still occurs
	// with multiple consumers, and gcs does not have transactions spanning several operations
	response, err := s.versions(ctx, &artifact.VersionsRequest{
		AppName: req.AppName, UserID: req.UserID, SessionID: req.SessionID, FileName: req.FileName,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list artifact versions: %w", err)
	}
	if len(response.Versions) > 0 {
		nextVersion = slices.Max(response.Versions) + 1
	}

	blobName := buildBlobName(appName, userID, sessionID, fileName, nextVersion)
	writer := s.bucket.object(blobName).newWriter(ctx)
	defer func() {
		if closeErr := writer.Close(); closeErr != nil && err == nil {
			err = fmt.Errorf("failed to close blob reader: %w", closeErr)
		}
	}()

	if newArtifact.InlineData != nil {
		writer.SetContentType(newArtifact.InlineData.MIMEType)
		if _, err := writer.Write(newArtifact.InlineData.Data); err != nil {
			return nil, fmt.Errorf("failed to write blob to GCS: %w", err)
		}
	} else {
		writer.SetContentType("text/plain")
		if _, err := writer.Write([]byte(newArtifact.Text)); err != nil {
			return nil, fmt.Errorf("failed to write text to GCS: %w", err)
		}
	}

	return &artifact.SaveResponse{Version: nextVersion}, nil
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
	version := req.Version

	if version == 0 {
		response, err := s.versions(ctx, &artifact.VersionsRequest{
			AppName: req.AppName, UserID: req.UserID, SessionID: req.SessionID, FileName: req.FileName,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to list artifact versions: %w", err)
		}
		if len(response.Versions) == 0 {
			return nil, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
		}
		version = slices.Max(response.Versions)
	}

	blobName := buildBlobName(appName, userID, sessionID, fileName, version)
	blob := s.bucket.object(blobName)

	// Check if the blob exists before trying to read it
	attrs, err := blob.attrs(ctx)
	if err != nil {
		if err == storage.ErrObjectNotExist {
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
		// TODO agent can create files with multiple segments for example file a/b.txt
		// This a/b.txt file will show as b.txt when listed and trying to load it will fail.
		filename := segments[len(segments)-2] // appName/userId/sessionId/filename/version or appName/userId/user/filename/version
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
