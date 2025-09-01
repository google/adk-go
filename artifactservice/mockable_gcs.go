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

package artifactservice

import (
	"bytes"
	"context"
	"io"
	"io/fs"
	"strings"
	"sync"
	"time"

	"cloud.google.com/go/storage"
	"google.golang.org/api/iterator"
)

// ------------------------ Defining interfaces to enable mocking --------------------------------
// GCSClient is an interface that a GCS client must satisfy.
type GCSClient interface {
	Bucket(name string) GCSBucket
}

// GCSBucket is an interface that a GCS bucket handle must satisfy.
type GCSBucket interface {
	Object(name string) GCSObject
	Objects(ctx context.Context, q *storage.Query) GCSObjectIterator
}

// GCSObject is an interface that a GCS object handle must satisfy.
type GCSObject interface {
	NewWriter(ctx context.Context) GCSWriter
	NewReader(ctx context.Context) (io.ReadCloser, error)
	Delete(ctx context.Context) error
	Attrs(ctx context.Context) (*storage.ObjectAttrs, error)
}

// GCSObjectIterator
type GCSObjectIterator interface {
	Next() (*storage.ObjectAttrs, error)
}

// GCSObjectWriter
type GCSWriter interface {
	io.Writer // Provides Write(p []byte) (n int, err error)
	io.Closer // Provides Close() error
	SetContentType(string)
}

// ---------------------- Wrapper Implementations for Real GCS Types --------------------------------
// GCSClientWrapper wraps a storage.Client to satisfy the GCSClient interface.
type GCSClientWrapper struct {
	client *storage.Client
}

// Bucket returns a GCSBucketWrapper that satisfies the GCSBucket interface.
func (w *GCSClientWrapper) Bucket(name string) GCSBucket {
	bucketHandle := w.client.Bucket(name)
	return &GCSBucketWrapper{bucket: bucketHandle}
}

// GCSBucketWrapper wraps a storage.BucketHandle to satisfy the GCSBucket interface.
type GCSBucketWrapper struct {
	bucket *storage.BucketHandle
}

// Object returns a GCSObjectWrapper that satisfies the GCSObject interface.
func (w *GCSBucketWrapper) Object(name string) GCSObject {
	objectHandle := w.bucket.Object(name)
	return &GCSObjectWrapper{object: objectHandle}
}

// Objects implements the GCSBucket interface for GCSBucketWrapper.
// It directly calls the underlying storage.BucketHandle's Objects method.
// The GCSBucketWrapper returns an implementation of the GCSObjectIterator interface.
func (w *GCSBucketWrapper) Objects(ctx context.Context, q *storage.Query) GCSObjectIterator {
	// This is the real GCS iterator.
	realIterator := w.bucket.Objects(ctx, q)
	// We return a wrapper around the real iterator.
	return &GCSObjectIteratorWrapper{iter: realIterator}
}

// GCSObjectWrapper wraps a storage.ObjectHandle to satisfy the GCSObject interface.
type GCSObjectWrapper struct {
	object *storage.ObjectHandle
}

// NewWriter implements the GCSObject interface for GCSObjectWrapper.
func (w *GCSObjectWrapper) NewWriter(ctx context.Context) GCSWriter {
	return &GCSWriterWrapper{w: w.object.NewWriter(ctx)}
}

// NewReader implements the GCSObject interface for GCSObjectWrapper.
func (w *GCSObjectWrapper) NewReader(ctx context.Context) (io.ReadCloser, error) {
	return w.object.NewReader(ctx)
}

// Delete implements the GCSObject interface for GCSObjectWrapper.
func (w *GCSObjectWrapper) Delete(ctx context.Context) error {
	return w.object.Delete(ctx)
}

// Attrs implements the GCSObject interface for GCSObjectWrapper.
func (w *GCSObjectWrapper) Attrs(ctx context.Context) (*storage.ObjectAttrs, error) {
	return w.object.Attrs(ctx)
}

// Create the wrapper for the real iterator.
type GCSObjectIteratorWrapper struct {
	iter *storage.ObjectIterator
}

func (w *GCSObjectIteratorWrapper) Next() (*storage.ObjectAttrs, error) {
	return w.iter.Next()
}

// GCSWriterWrapper wraps the real GCS writer to satisfy our ObjectWriter interface.
type GCSWriterWrapper struct {
	w *storage.Writer
}

func (g *GCSWriterWrapper) Write(p []byte) (n int, err error) {
	return g.w.Write(p)
}

func (g *GCSWriterWrapper) Close() error {
	return g.w.Close()
}

func (g *GCSWriterWrapper) SetContentType(cType string) {
	g.w.ContentType = cType
}

var _ GCSClient = (*GCSClientWrapper)(nil)
var _ GCSBucket = (*GCSBucketWrapper)(nil)
var _ GCSObject = (*GCSObjectWrapper)(nil)
var _ GCSObjectIterator = (*GCSObjectIteratorWrapper)(nil)
var _ GCSWriter = (*GCSWriterWrapper)(nil)

// ---------------------------------- Mock Implementations -----------------------------------
// FakeClient implements the GCSClient interface for testing.
type FakeClient struct {
	inMemoryBucket GCSBucket
}

func NewFakeClient() GCSClient {
	return &FakeClient{
		inMemoryBucket: &FakeBucket{
			objects: make(map[string]*FakeObject),
		},
	}
}

// Bucket returns the singleton in-memory bucket.
func (c *FakeClient) Bucket(name string) GCSBucket {
	return c.inMemoryBucket
}

// FakeBucket implements the GCSBucket interface for testing.
type FakeBucket struct {
	mu      sync.Mutex
	objects map[string]*FakeObject
}

// Object returns a fake object from the in-memory store.
func (f *FakeBucket) Object(name string) GCSObject {
	f.mu.Lock()
	defer f.mu.Unlock()
	if _, ok := f.objects[name]; !ok {
		f.objects[name] = &FakeObject{name: name}
	}
	return f.objects[name]
}

// Objects simulates iterating over objects with a prefix.
func (f *FakeBucket) Objects(ctx context.Context, q *storage.Query) GCSObjectIterator {
	f.mu.Lock()
	defer f.mu.Unlock()

	var matchingObjects []*FakeObject
	for name, obj := range f.objects {
		if q != nil && q.Prefix != "" && !strings.HasPrefix(name, q.Prefix) {
			continue
		}
		if !obj.deleted {
			matchingObjects = append(matchingObjects, obj)
		}
	}

	// This is the key change. We return a custom type that has a `Next` method
	// that manages its own state and returns the correct values.
	return &FakeObjectIterator{
		objects: matchingObjects,
		index:   0,
	}
}

// FakeObject implements the GCSObject interface for testing.
type FakeObject struct {
	mu          sync.Mutex
	name        string
	data        []byte
	deleted     bool
	contentType string
}

// NewWriter returns a fake writer that stores data in memory.
func (f *FakeObject) NewWriter(ctx context.Context) GCSWriter {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.deleted = false // A write operation "undeletes" the object
	f.data = nil      // Clear existing data
	return &fakeWriter{obj: f, buffer: &bytes.Buffer{}}
}

// Attrs returns fake attributes for the object.
func (f *FakeObject) Attrs(ctx context.Context) (*storage.ObjectAttrs, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.deleted || f.data == nil {
		return nil, storage.ErrObjectNotExist
	}
	return &storage.ObjectAttrs{Name: f.name, Created: time.Now(), ContentType: f.contentType}, nil
}

// Delete marks the object as deleted in memory.
func (f *FakeObject) Delete(ctx context.Context) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.deleted = true
	return nil
}

// NewReader returns a reader for the in-memory data.
func (f *FakeObject) NewReader(ctx context.Context) (io.ReadCloser, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.deleted || f.data == nil {
		return nil, fs.ErrNotExist
	}
	return io.NopCloser(bytes.NewReader(f.data)), nil
}

// fakeWriter is a helper type to simulate an *storage.Writer
type fakeWriter struct {
	obj         *FakeObject
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

// FakeObjectIterator is a fake iterator that returns attributes from a slice.
// This type is the key to solving the 'unknown field' error.
type FakeObjectIterator struct {
	objects []*FakeObject
	index   int
}

// Next implements the iterator pattern.
// It returns the next object in the slice or an iterator.Done error.
func (i *FakeObjectIterator) Next() (*storage.ObjectAttrs, error) {
	if i.index >= len(i.objects) {
		return nil, iterator.Done
	}
	obj := i.objects[i.index]
	i.index++
	return &storage.ObjectAttrs{Name: obj.name, ContentType: obj.contentType}, nil
}

var _ GCSClient = (*FakeClient)(nil)
var _ GCSBucket = (*FakeBucket)(nil)
var _ GCSObject = (*FakeObject)(nil)
var _ GCSObjectIterator = (*FakeObjectIterator)(nil)
var _ GCSWriter = (*fakeWriter)(nil)
