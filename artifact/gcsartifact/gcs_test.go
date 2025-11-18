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
	"bytes"
	"context"
	"io"
	"io/fs"
	"strings"
	"sync"
	"testing"
	"time"

	"cloud.google.com/go/storage"
	"google.golang.org/api/iterator"

	"google.golang.org/adk/artifact"
	"google.golang.org/adk/internal/artifact/tests"
)

// newGCSArtifactServiceForTesting creates a gcsService for the specified bucket using a mocked inmemory client
func newGCSArtifactServiceForTesting(bucketName string) (artifact.Service, error) {
	client := newFakeClient()
	s := &gcsService{
		bucketName:    bucketName,
		storageClient: client,
		bucket:        client.bucket(bucketName),
	}
	return s, nil
}

func TestGCSArtifactService(t *testing.T) {
	factory := func(t *testing.T) (artifact.Service, error) {
		return newGCSArtifactServiceForTesting("new")
	}
	tests.TestArtifactService(t, "GCS", factory)
}

// ---------------------------------- Mock Implementations -----------------------------------
// fakeClient implements the gcsClient interface for testing.
type fakeClient struct {
	inMemoryBucket gcsBucket
}

func newFakeClient() gcsClient {
	return &fakeClient{
		inMemoryBucket: &fakeBucket{
			objectsMap: make(map[string]*fakeObject),
		},
	}
}

// Bucket returns the singleton in-memory bucket.
func (c *fakeClient) bucket(name string) gcsBucket {
	return c.inMemoryBucket
}

// fakeBucket implements the gcsBucket interface for testing.
type fakeBucket struct {
	mu         sync.Mutex
	objectsMap map[string]*fakeObject
}

// Object returns a fake object from the in-memory store.
func (f *fakeBucket) object(name string) gcsObject {
	f.mu.Lock()
	defer f.mu.Unlock()
	if _, ok := f.objectsMap[name]; !ok {
		f.objectsMap[name] = &fakeObject{name: name}
	}
	return f.objectsMap[name]
}

// Objects simulates iterating over objects with a prefix.
func (f *fakeBucket) objects(ctx context.Context, q *storage.Query) gcsObjectIterator {
	f.mu.Lock()
	defer f.mu.Unlock()

	var matchingObjects []*fakeObject
	for name, obj := range f.objectsMap {
		if q != nil && q.Prefix != "" && !strings.HasPrefix(name, q.Prefix) {
			continue
		}
		if !obj.deleted {
			matchingObjects = append(matchingObjects, obj)
		}
	}

	// This is the key change. We return a custom type that has a `Next` method
	// that manages its own state and returns the correct values.
	return &fakeObjectIterator{
		objects: matchingObjects,
		index:   0,
	}
}

// fakeObject implements the gcsObject interface for testing.
type fakeObject struct {
	mu          sync.Mutex
	name        string
	data        []byte
	deleted     bool
	contentType string
}

// NewWriter returns a fake writer that stores data in memory.
func (f *fakeObject) newWriter(ctx context.Context) gcsWriter {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.deleted = false // A write operation "undeletes" the object
	f.data = nil      // Clear existing data
	return &fakeWriter{obj: f, buffer: &bytes.Buffer{}}
}

// Attrs returns fake attributes for the object.
func (f *fakeObject) attrs(ctx context.Context) (*storage.ObjectAttrs, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.deleted || f.data == nil {
		return nil, storage.ErrObjectNotExist
	}
	return &storage.ObjectAttrs{Name: f.name, Created: time.Now(), ContentType: f.contentType}, nil
}

// Delete marks the object as deleted in memory.
func (f *fakeObject) delete(ctx context.Context) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.deleted = true
	return nil
}

// NewReader returns a reader for the in-memory data.
func (f *fakeObject) newReader(ctx context.Context) (io.ReadCloser, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.deleted || f.data == nil {
		return nil, fs.ErrNotExist
	}
	return io.NopCloser(bytes.NewReader(f.data)), nil
}

// fakeWriter is a helper type to simulate an *storage.Writer
type fakeWriter struct {
	obj         *fakeObject
	buffer      *bytes.Buffer
	contentType string
}

func (w *fakeWriter) Write(p []byte) (n int, err error) {
	return w.buffer.Write(p)
}

func (w *fakeWriter) Close() error {
	w.obj.mu.Lock()
	defer w.obj.mu.Unlock()
	w.obj.data = w.buffer.Bytes()
	w.obj.contentType = w.contentType
	return nil
}

// SetContentType implements the final piece of the interface.
func (w *fakeWriter) SetContentType(cType string) {
	w.contentType = cType
}

// fakeObjectIterator is a fake iterator that returns attributes from a slice.
// This type is the key to solving the 'unknown field' error.
type fakeObjectIterator struct {
	objects []*fakeObject
	index   int
}

// Next implements the iterator pattern.
// It returns the next object in the slice or an iterator.Done error.
func (i *fakeObjectIterator) next() (*storage.ObjectAttrs, error) {
	if i.index >= len(i.objects) {
		return nil, iterator.Done
	}
	obj := i.objects[i.index]
	i.index++
	return &storage.ObjectAttrs{Name: obj.name, ContentType: obj.contentType}, nil
}

var (
	_ gcsClient         = (*fakeClient)(nil)
	_ gcsBucket         = (*fakeBucket)(nil)
	_ gcsObject         = (*fakeObject)(nil)
	_ gcsObjectIterator = (*fakeObjectIterator)(nil)
	_ gcsWriter         = (*fakeWriter)(nil)
)
