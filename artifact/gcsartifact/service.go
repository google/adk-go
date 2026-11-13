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
//
// Each object stores one artifact version. Besides the caller's custom
// metadata, an object carries bookkeeping metadata keys that the service
// reserves, in the format adk-python's GCS artifact service uses:
// "adkIsText" marks a text part, "adkDisplayName" holds the display name of
// inline data, and "adkFileUri" and "adkFileMimeType" (or the legacy
// "file_uri") describe a file-data part, which is stored without a payload.
// A file-data part whose URI uses the artifact:// scheme is a reference to
// another artifact, which Load and the URL calls follow.
//
// Version numbers differ from adk-python's: this service numbers an
// artifact's versions from 1, and adk-python from 0. A reference that names
// version 0, as adk-python writes for a first version, is refused with an
// error wrapping [artifact.ErrInputValidation].
//
// Load returns a version that holds the rewind marker
// ([artifact.IsRewindTombstone]) as an empty "application/octet-stream" part;
// it does not report it as not found, as the in-memory service does. Versions
// returns version numbers in ascending order.
//
// The service returned by [NewService] also implements [URLProvider], for
// authenticated and signed URLs of stored artifacts.
package gcsartifact

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"math/rand/v2"
	"net/http"
	"slices"
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
	"google.golang.org/adk/v2/artifact/internal/pathquote"
	"google.golang.org/adk/v2/platform"
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
// The returned service also implements [URLProvider]; reach it with a type
// assertion:
//
//	svc, err := gcsartifact.NewService(ctx, "my-bucket")
//	...
//	url, err := svc.(gcsartifact.URLProvider).GetAuthenticatedURL(ctx, req)
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

// URLProvider provides direct HTTP URL access to stored artifacts.
// It is implemented by services (like GCS) that can produce direct URLs to artifacts.
// The caller must provide valid URL requests. It guarantees that the URL string returned points to the artifact.
// Safe for concurrent use.
//
// Both calls follow artifact:// references the way Load does, and return the
// URL of the artifact the chain ends at. A storage error other than "object
// not found" is returned wrapped.
type URLProvider interface {
	// GetAuthenticatedURL returns an authenticated URL for the artifact.
	// Caller must provide a non-nil request. Guarantees a percent-encoded https:// URL string on success.
	// If the artifact does not exist or refers to external file data, it returns an empty string and a nil error.
	// On validation or recursion failure, returns an error wrapping artifact.ErrInputValidation.
	// Safe for concurrent use.
	//
	// The URL is https://storage.cloud.google.com/{bucket}/{object}, with the
	// object name percent-encoded except for "/" and the characters that never
	// need it. Opening it requires a Google account that may read the object.
	GetAuthenticatedURL(ctx context.Context, req *GetAuthenticatedURLRequest) (string, error)

	// GetSignedURL returns a time-limited signed URL for the artifact.
	// Caller must provide a non-nil request. Guarantees a signed URL string on success.
	// If the artifact does not exist or refers to external file data, it returns an empty string and a nil error.
	// On validation, recursion, or signing failure, returns an error.
	// Safe for concurrent use.
	//
	// Validation and recursion errors wrap artifact.ErrInputValidation, and a
	// signer error is returned wrapped. The signer is called only when a URL
	// is returned. GetSignedURL does not modify req.
	GetSignedURL(ctx context.Context, req *GetSignedURLRequest) (string, error)
}

// GetAuthenticatedURLRequest contains the parameters for an authenticated URL lookup.
// Caller must provide valid AppName, UserID, and FileName. SessionID is required unless FileName starts with "user:".
// The zero value (nil Version) requests the latest version of an empty filename (which will fail validation).
//
// The fields follow the rules of [artifact.LoadRequest.Validate].
type GetAuthenticatedURLRequest struct {
	AppName   string
	UserID    string
	SessionID string
	FileName  string
	Version   *int64 // nil means the latest version.
}

// URLSigner signs a Cloud Storage URL.
// Caller must provide a valid bucket name, object name, and options. Guarantees a signed URL.
// On failure, returns an error. The zero value is nil and cannot be called.
// Safe for concurrent use.
//
// GetSignedURL passes the bucket of the service, the raw (not
// percent-encoded) name of the object after version choice and reference
// following, and options with Method, Expires and Scheme set from the
// request. The signer may modify opts; each call gets its own.
type URLSigner func(bucket, object string, opts *storage.SignedURLOptions) (string, error)

// GetSignedURLRequest contains parameters for generating a signed URL.
// Caller must provide valid app/user/file scope.
// The zero value (nil Version) requests the latest version of an empty filename with a default 1-hour Expiration, "GET" Method, and the storage library's default Scheme.
//
// The scope fields follow the rules of [artifact.LoadRequest.Validate].
type GetSignedURLRequest struct {
	AppName   string
	UserID    string
	SessionID string
	FileName  string
	Version   *int64 // nil means the latest version.

	// Expiration is the lifetime of the signed URL. If zero, defaults to 1 hour.
	Expiration time.Duration
	// Method is the HTTP method. If empty, defaults to "GET".
	Method string
	// Scheme is the signing scheme. If zero, defaults to the storage library's default (V2).
	// The zero value, storage.SigningSchemeDefault, reaches the signer
	// unchanged, meaning that no signing version was asked for.
	Scheme storage.SigningScheme

	// Signer is an optional custom signing function. If nil, the default mechanism is used.
	// It belongs on this request (rather than the service) because a production user may need
	// to sign URLs using different service accounts or key services depending on the specific request
	// (e.g. per-user or per-file dynamic credentials), preventing it from being fixed at the service level.
	// Example production use:
	//
	//	Signer: func(bucket, object string, opts *storage.SignedURLOptions) (string, error) {
	//		// use custom IAM credentials or key service for this specific object
	//		return myKeyService.Sign(bucket, object, opts)
	//	}
	//
	// The default mechanism is [storage.BucketHandle.SignedURL] on the client
	// NewService built, which finds the signing identity and key in the
	// client's credentials, or signs through the IAM signBlob API.
	Signer URLSigner
}

// GetAuthenticatedURL implements [URLProvider].
func (s *gcsService) GetAuthenticatedURL(ctx context.Context, req *GetAuthenticatedURLRequest) (string, error) {
	blobName, err := s.urlTarget(ctx, artifactID{
		appName: req.AppName, userID: req.UserID, sessionID: req.SessionID, fileName: req.FileName,
	}, req.Version)
	if err != nil || blobName == "" {
		return "", err
	}
	return "https://storage.cloud.google.com/" + s.bucketName + "/" + pathquote.Quote(blobName), nil
}

// GetSignedURL implements [URLProvider].
func (s *gcsService) GetSignedURL(ctx context.Context, req *GetSignedURLRequest) (string, error) {
	blobName, err := s.urlTarget(ctx, artifactID{
		appName: req.AppName, userID: req.UserID, sessionID: req.SessionID, fileName: req.FileName,
	}, req.Version)
	if err != nil || blobName == "" {
		return "", err
	}
	opts := &storage.SignedURLOptions{
		Method:  cmp.Or(req.Method, http.MethodGet),
		Expires: platform.Now(ctx).Add(cmp.Or(req.Expiration, time.Hour)),
		Scheme:  req.Scheme,
	}
	var url string
	if req.Signer != nil {
		url, err = req.Signer(s.bucketName, blobName, opts)
	} else {
		url, err = s.bucket.signedURL(blobName, opts)
	}
	if err != nil {
		return "", fmt.Errorf("failed to sign artifact URL: %w", err)
	}
	return url, nil
}

// urlTarget returns the name of the object a URL call points at: version, or
// the latest version when version is nil, of the artifact id names, after
// following references. It returns "" and a nil error when that version does
// not exist or holds external file data, as adk-python returns None for both.
func (s *gcsService) urlTarget(ctx context.Context, id artifactID, version *int64) (string, error) {
	if err := id.validate(); err != nil {
		return "", fmt.Errorf("request validation failed: %w", err)
	}
	var err error
	if version != nil {
		id.version = *version
	} else if id.version, err = s.resolveVersion(ctx, id.appName, id.userID, id.sessionID, id.fileName, 0); err != nil {
		return "", notFoundAsEmpty(err)
	}
	id, attrs, err := s.resolveObject(ctx, id)
	if err != nil {
		return "", notFoundAsEmpty(err)
	}
	if storedFileURI(attrs.Metadata) != "" {
		return "", nil
	}
	return id.blobName(), nil
}

// notFoundAsEmpty returns nil for an error that wraps fs.ErrNotExist and err
// otherwise.
func notFoundAsEmpty(err error) error {
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	return err
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
	// Session ID "user" shares this path with user-scoped artifacts, so include
	// the filename namespace to exclude its private files. Python rejects the
	// ID instead, but Go already accepts it.
	return fmt.Sprintf("%s/%s/user/user:", appName, userID)
}

// The blob metadata keys the service writes for its own bookkeeping, the
// same as adk-python's GCS artifact service. Save drops caller metadata under
// these names and GetArtifactVersion hides them.
const (
	displayNameMetadataKey   = "adkDisplayName"
	isTextMetadataKey        = "adkIsText"
	fileURIMetadataKey       = "adkFileUri"
	fileMIMETypeMetadataKey  = "adkFileMimeType"
	legacyFileURIMetadataKey = "file_uri"
)

var reservedMetadataKeys = []string{
	displayNameMetadataKey, isTextMetadataKey, fileURIMetadataKey, fileMIMETypeMetadataKey, legacyFileURIMetadataKey,
}

// artifactURIScheme is the scheme of a file-data URI that references another
// artifact.
const artifactURIScheme = "artifact://"

// storesFileData reports whether Save stores part as file data. Text wins
// over file data, as in adk-python, but genai.Part cannot tell empty text
// from unset text, so a part with empty text and file data is file data.
func storesFileData(part *genai.Part) bool {
	return part.InlineData == nil && part.Text == "" && part.FileData != nil
}

// blobMetadata returns the metadata Save stores for part: the caller's
// values converted with fmt.Sprint, except those under a bookkeeping key,
// and the bookkeeping keys Load rebuilds part from. It returns nil when
// there is nothing to store.
func blobMetadata(part *genai.Part, custom map[string]any) map[string]string {
	metadata := make(map[string]string, len(custom)+2)
	for key, value := range custom {
		if !slices.Contains(reservedMetadataKeys, key) {
			metadata[key] = fmt.Sprint(value)
		}
	}
	switch {
	case part.InlineData != nil:
		if part.InlineData.DisplayName != "" {
			metadata[displayNameMetadataKey] = part.InlineData.DisplayName
		}
	case storesFileData(part):
		metadata[fileURIMetadataKey] = part.FileData.FileURI
		if part.FileData.MIMEType != "" {
			metadata[fileMIMETypeMetadataKey] = part.FileData.MIMEType
		}
	default:
		metadata[isTextMetadataKey] = "true"
	}
	if len(metadata) == 0 {
		return nil
	}
	return metadata
}

// storedFileURI returns the file URI an object records, under the current
// key or, when that is missing or empty, the legacy one.
func storedFileURI(metadata map[string]string) string {
	return cmp.Or(metadata[fileURIMetadataKey], metadata[legacyFileURIMetadataKey])
}

// partFromObject rebuilds the part Save stored in an object with attrs and
// payload data, in adk-python's order: file data, then text, then inline data
// with a display name, then plain inline data.
func partFromObject(attrs *storage.ObjectAttrs, data []byte) *genai.Part {
	metadata := attrs.Metadata
	if uri := storedFileURI(metadata); uri != "" {
		// A present but empty MIME type key means no MIME type; only a missing
		// key falls back to the content type.
		mimeType, ok := metadata[fileMIMETypeMetadataKey]
		if !ok {
			mimeType = attrs.ContentType
		}
		return &genai.Part{FileData: &genai.FileData{FileURI: uri, MIMEType: mimeType}}
	}
	if metadata[isTextMetadataKey] == "true" {
		return genai.NewPartFromText(string(data))
	}
	if name := metadata[displayNameMetadataKey]; name != "" {
		return &genai.Part{InlineData: &genai.Blob{Data: data, MIMEType: attrs.ContentType, DisplayName: name}}
	}
	return genai.NewPartFromBytes(data, attrs.ContentType)
}

// artifactID names one version of an artifact.
type artifactID struct {
	appName, userID, sessionID, fileName string
	version                              int64
}

func (id artifactID) blobName() string {
	return buildBlobName(id.appName, id.userID, id.sessionID, id.fileName, id.version)
}

// validate checks the names in id with the rules of a load request.
func (id artifactID) validate() error {
	return (&artifact.LoadRequest{
		AppName: id.appName, UserID: id.userID, SessionID: id.sessionID, FileName: id.fileName,
	}).Validate()
}

// resolveObject returns the attributes of the object id names, following
// artifact references, and the id of the object the chain ends at. A hop
// reads exactly the version its reference names. It checks each reference
// with [artifact.ParseAndValidateReferenceURI] in the scope of the artifact
// holding it, and validates the target as a load request, as adk-python
// does for every hop. A missing object returns an error wrapping
// fs.ErrNotExist.
func (s *gcsService) resolveObject(ctx context.Context, id artifactID) (artifactID, *storage.ObjectAttrs, error) {
	for remainingDepth := artifact.MaxArtifactReferenceDepth; ; remainingDepth-- {
		blobName := id.blobName()
		attrs, err := s.bucket.object(blobName).attrs(ctx)
		if err != nil {
			if errors.Is(err, storage.ErrObjectNotExist) {
				return artifactID{}, nil, fmt.Errorf("artifact '%s' not found: %w", blobName, fs.ErrNotExist)
			}
			return artifactID{}, nil, fmt.Errorf("could not get blob attributes: %w", err)
		}
		uri := storedFileURI(attrs.Metadata)
		if !strings.HasPrefix(uri, artifactURIScheme) {
			return id, attrs, nil
		}
		target, err := artifact.ParseAndValidateReferenceURI(uri, id.appName, id.userID, id.sessionID, remainingDepth)
		if err != nil {
			return artifactID{}, nil, err
		}
		id = artifactID{
			appName: target.AppName, userID: target.UserID, sessionID: target.SessionID,
			fileName: target.FileName, version: target.Version,
		}
		if err := id.validate(); err != nil {
			return artifactID{}, nil, fmt.Errorf("invalid artifact reference target: %w", err)
		}
	}
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
//
// Save accepts file data (see [artifact.SaveRequest.ValidateAllowingFileData])
// and stores it as metadata on an object without a payload. An artifact://
// reference must be well formed and stay within the caller's scope; Save
// refuses it with an error wrapping [artifact.ErrInputValidation] before
// anything is listed or written. Caller metadata under a bookkeeping key
// (see the package documentation) is dropped.
func (s *gcsService) Save(ctx context.Context, req *artifact.SaveRequest) (*artifact.SaveResponse, error) {
	if err := req.ValidateAllowingFileData(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	appName, userID, sessionID, fileName := req.AppName, req.UserID, req.SessionID, req.FileName
	if storesFileData(req.Part) && strings.HasPrefix(req.Part.FileData.FileURI, artifactURIScheme) {
		if _, err := artifact.ParseAndValidateReferenceURI(req.Part.FileData.FileURI, appName, userID, sessionID, artifact.MaxArtifactReferenceDepth); err != nil {
			return nil, fmt.Errorf("request validation failed: %w", err)
		}
	}
	customMetadata := blobMetadata(req.Part, req.CustomMetadata)

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
		err = writeArtifact(ctx, obj, req.Part, customMetadata)
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
func writeArtifact(ctx context.Context, obj gcsObject, part *genai.Part, metadata map[string]string) (err error) {
	writer := obj.newWriter(ctx)
	defer func() {
		if closeErr := writer.Close(); closeErr != nil && err == nil {
			err = fmt.Errorf("failed to close blob writer: %w", closeErr)
		}
	}()
	writer.SetMetadata(metadata)

	switch {
	case part.InlineData != nil:
		writer.SetContentType(part.InlineData.MIMEType)
		if _, err := writer.Write(part.InlineData.Data); err != nil {
			return fmt.Errorf("failed to write blob to GCS: %w", err)
		}
	case storesFileData(part):
		// The metadata holds the file URI; the object has no payload.
		writer.SetContentType(part.FileData.MIMEType)
	default:
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
	if apiErr, ok := errors.AsType[*googleapi.Error](err); ok {
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
	for _, v := range response.Versions {
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

// Load implements [artifact.Service].
//
// Load rebuilds the part from the object's bookkeeping metadata: file data,
// text, or inline data with its display name. It follows artifact://
// references to the version each names, at most
// [artifact.MaxArtifactReferenceDepth] hops, and checks every hop's scope; a
// cycle, a malformed reference or one leaving the scope returns an error
// wrapping [artifact.ErrInputValidation]. A missing target returns an error
// wrapping fs.ErrNotExist.
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

	id, attrs, err := s.resolveObject(ctx, artifactID{
		appName: appName, userID: userID, sessionID: sessionID, fileName: fileName, version: version,
	})
	if err != nil {
		return nil, err
	}
	if storedFileURI(attrs.Metadata) != "" {
		return &artifact.LoadResponse{Part: partFromObject(attrs, nil)}, nil
	}
	data, err := s.readObject(ctx, id.blobName())
	if err != nil {
		return nil, err
	}
	return &artifact.LoadResponse{Part: partFromObject(attrs, data)}, nil
}

// readObject returns the payload of the named object.
func (s *gcsService) readObject(ctx context.Context, blobName string) (_ []byte, err error) {
	reader, err := s.bucket.object(blobName).newReader(ctx)
	if err != nil {
		return nil, fmt.Errorf("could not create reader for blob '%s': %w", blobName, err)
	}
	defer func() {
		if closeErr := reader.Close(); closeErr != nil && err == nil {
			err = fmt.Errorf("failed to close blob reader: %w", closeErr)
		}
	}()
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("could not read data from blob '%s': %w", blobName, err)
	}
	return data, nil
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

	// An empty session asks for user-scoped names only. Its prefix would be
	// "app/user//", which also matches the blobs of a user ID ending in "/".
	if sessionID != "" {
		err = s.fetchFilenamesFromPrefix(ctx, buildSessionPrefix(appName, userID, sessionID), filenamesSet)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch session filenames: %w", err)
		}
	}

	// Fetch filenames for the user.
	err = s.fetchFilenamesFromPrefix(ctx, buildUserPrefix(appName, userID), filenamesSet)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch user filenames: %w", err)
	}

	filenames := slices.Sorted(maps.Keys(filenamesSet))
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
		if version, ok := parseVersion(blob.Name, prefix); ok {
			versions = append(versions, version)
		}
	}
	slices.Sort(versions)
	return &artifact.VersionsResponse{Versions: versions}, nil
}

// parseVersion returns the version that the object blobName holds of the
// artifact whose object names start with prefix, as adk-python's
// _parse_version. The rest of the name must be ASCII digits: a name with a
// further "/" belongs to another artifact, and any other name, such as one
// with a sign, is not an object this service wrote.
func parseVersion(blobName, prefix string) (int64, bool) {
	suffix, ok := strings.CutPrefix(blobName, prefix)
	if !ok || suffix == "" {
		return 0, false
	}
	for i := 0; i < len(suffix); i++ {
		if suffix[i] < '0' || suffix[i] > '9' {
			return 0, false
		}
	}
	version, err := strconv.ParseInt(suffix, 10, 64)
	return version, err == nil
}

// Versions implements [artifact.Service] and returns an error if no versions are found.
// The versions are in ascending order.
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
// CustomMetadata leaves out the bookkeeping keys. GetArtifactVersion does not
// follow a reference: it describes the version that holds it.
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

	// Always use the gs:// form, matching adk-python's GCS artifact service. The
	// object's MediaLink is an authenticated JSON API download URL, which a
	// consumer handed the URI cannot fetch: a model given it as the file_uri of
	// a file_data part treats it as a web page and fails to read it. Keep the
	// object name literal so the URI names the object Save wrote, including any
	// spaces or non-ASCII characters.
	canonicalURI := fmt.Sprintf("gs://%s/%s", s.bucketName, blobName)

	customMeta := make(map[string]any, len(attrs.Metadata))
	for k, v := range attrs.Metadata {
		if !slices.Contains(reservedMetadataKeys, k) {
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
