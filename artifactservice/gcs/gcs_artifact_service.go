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
	as "google.golang.org/adk/artifactservice"
	"google.golang.org/api/iterator"
	"google.golang.org/genai"
)

// gcsService is an google cloud storage implementation of the Service.
type gcsService struct {
	bucketName    string
	storageClient gcsClient
	bucket        gcsBucket
}

// NewGCSArtifactService creates a gcsService for the specified bucket using a default client
func NewGCSArtifactService(ctx context.Context, bucketName string) (as.Service, error) {
	var err error
	storageClient, err := storage.NewClient(ctx)
	if err != nil {
		return nil, err
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

// newGCSArtifactServiceForTesting creates a gcsService for the specified bucket using a mocked inmemory client
func newGCSArtifactServiceForTesting(ctx context.Context, bucketName string) (as.Service, error) {
	client := newFakeClient()
	s := &gcsService{
		bucketName:    bucketName,
		storageClient: client,
		bucket:        client.bucket(bucketName),
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

func (s *gcsService) Save(ctx context.Context, req *as.SaveRequest) (_ *as.SaveResponse, err error) {
	err = req.Validate()
	if err != nil {
		return nil, err
	}
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	artifact := req.Part
	if artifact.InlineData == nil {
		return nil, fmt.Errorf("failed to save to GCS: Part.InlineData cannot be nil")
	}

	nextVersion := int64(1)

	// TODO race condition, could use mutex but it's a remote resource so the issue would still occurs
	// with multiple consumers, and gcs does not have transactions spanning several operations
	response, err := s.versions(ctx, &as.VersionsRequest{
		AppName: req.AppName, UserID: req.UserID, SessionID: req.SessionID, FileName: req.FileName,
	})
	if err != nil {
		return nil, err
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

	writer.SetContentType(artifact.InlineData.MIMEType)

	if _, err := writer.Write(artifact.InlineData.Data); err != nil {
		return nil, fmt.Errorf("failed to write to GCS: %w", err)
	}

	return &as.SaveResponse{Version: nextVersion}, nil
}

func (s *gcsService) Delete(ctx context.Context, req *as.DeleteRequest) error {
	err := req.Validate()
	if err != nil {
		return err
	}
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	version := req.Version

	//Delete specific version
	if version != 0 {
		blobName := buildBlobName(appName, userID, sessionID, fileName, version)
		obj := s.bucket.object(blobName)
		return obj.delete(ctx)
	}

	// Delete all versions
	response, err := s.versions(ctx, &as.VersionsRequest{
		AppName: req.AppName, UserID: req.UserID, SessionID: req.SessionID, FileName: req.FileName,
	})
	if err != nil {
		return err
	}
	for _, version := range response.Versions {
		blobName := buildBlobName(appName, userID, sessionID, fileName, version)
		// Delete the object using its full name
		obj := s.bucket.object(blobName)
		if err := obj.delete(ctx); err != nil {
			return fmt.Errorf("failed to delete object %s: %w", blobName, err)
		}
	}

	return nil
}

func (s *gcsService) Load(ctx context.Context, req *as.LoadRequest) (_ *as.LoadResponse, err error) {
	err = req.Validate()
	if err != nil {
		return nil, err
	}
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	version := req.Version

	if version <= 0 {
		response, err := s.versions(ctx, &as.VersionsRequest{
			AppName: req.AppName, UserID: req.UserID, SessionID: req.SessionID, FileName: req.FileName,
		})
		if err != nil {
			return nil, err
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

	return &as.LoadResponse{Part: part}, nil
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
			return fmt.Errorf("error iterating blobs: incorrect number of segments in path")
		}
		// TODO agent can create files with multiple segments for example file a/b.txt
		// This a/b.txt file will show as b.txt when listed and trying to load it will fail.
		filename := segments[len(segments)-2] // appName/userId/sessionId/filename/version or appName/userId/user/filename/version
		filenamesSet[filename] = true
	}

	return nil
}

func (s *gcsService) List(ctx context.Context, req *as.ListRequest) (*as.ListResponse, error) {
	err := req.Validate()
	if err != nil {
		return nil, err
	}
	appName, userID, sessionID := req.AppName, req.UserID, req.SessionID
	filenamesSet := map[string]bool{}

	// Fetch filenames for the session.
	err = s.fetchFilenamesFromPrefix(ctx, buildSessionPrefix(appName, userID, sessionID), filenamesSet)
	if err != nil {
		return nil, err
	}

	// Fetch filenames for the user.
	err = s.fetchFilenamesFromPrefix(ctx, buildUserPrefix(appName, userID), filenamesSet)
	if err != nil {
		return nil, err
	}

	filenames := slices.Collect(maps.Keys(filenamesSet))
	sort.Strings(filenames)
	return &as.ListResponse{FileNames: filenames}, nil
}

// versions internal function that does not return error if versions are empty
func (s *gcsService) versions(ctx context.Context, req *as.VersionsRequest) (*as.VersionsResponse, error) {
	err := req.Validate()
	if err != nil {
		return nil, err
	}
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName

	prefix := buildBlobNamePrefix(appName, userID, sessionID, fileName)
	query := &storage.Query{
		Prefix: prefix,
	}
	blobsIterator := s.bucket.objects(ctx, query)

	var versions = make([]int64, 0)
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
			return nil, fmt.Errorf("error iterating blobs: incorrect number of segments in path")
		}
		version, err := strconv.ParseInt(segments[len(segments)-1], 10, 64)
		//if the file version is not convertable to number, just ignore it
		if err != nil {
			continue
		}
		versions = append(versions, version)
	}
	return &as.VersionsResponse{Versions: versions}, nil
}

// Versions implements types.Service and return err if versions is empty
func (s *gcsService) Versions(ctx context.Context, req *as.VersionsRequest) (*as.VersionsResponse, error) {
	response, err := s.versions(ctx, req)
	if err != nil {
		return response, err
	}
	if len(response.Versions) == 0 {
		return nil, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
	}
	return response, err
}

// Ensure interface implementation
var _ as.Service = (*gcsService)(nil)
