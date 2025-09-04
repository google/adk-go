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
	"time"

	"cloud.google.com/go/storage"
	"google.golang.org/api/iterator"
)

// ------------------------ Defining interfaces to enable mocking --------------------------------
// gcsClient is an interface that a gcs client must satisfy.
type gcsClient interface {
	bucket(name string) gcsBucket
}

// gcsBucket is an interface that a gcs bucket handle must satisfy.
type gcsBucket interface {
	object(name string) gcsObject
	objects(ctx context.Context, q *storage.Query) gcsObjectIterator
}

// gcsObject is an interface that a gcs object handle must satisfy.
type gcsObject interface {
	newWriter(ctx context.Context) gcsWriter
	newReader(ctx context.Context) (io.ReadCloser, error)
	delete(ctx context.Context) error
	attrs(ctx context.Context) (*storage.ObjectAttrs, error)
}

// gcsObjectIterator
type gcsObjectIterator interface {
	next() (*storage.ObjectAttrs, error)
}

// gcsObjectWriter
type gcsWriter interface {
	io.Writer // Provides Write(p []byte) (n int, err error)
	io.Closer // Provides Close() error
	SetContentType(string)
}

// ---------------------- Wrapper Implementations for Real gcs Types --------------------------------
// gcsClientWrapper wraps a storage.Client to satisfy the gcsClient interface.
type gcsClientWrapper struct {
	client *storage.Client
}

// Bucket returns a gcsBucketWrapper that satisfies the gcsBucket interface.
func (w *gcsClientWrapper) bucket(name string) gcsBucket {
	bucketHandle := w.client.Bucket(name)
	return &gcsBucketWrapper{bucket: bucketHandle}
}

// gcsBucketWrapper wraps a storage.BucketHandle to satisfy the gcsBucket interface.
type gcsBucketWrapper struct {
	bucket *storage.BucketHandle
}

// Object returns a gcsObjectWrapper that satisfies the gcsObject interface.
func (w *gcsBucketWrapper) object(name string) gcsObject {
	objectHandle := w.bucket.Object(name)
	return &gcsObjectWrapper{object: objectHandle}
}

// Objects implements the gcsBucket interface for gcsBucketWrapper.
// It directly calls the underlying storage.BucketHandle's Objects method.
// The gcsBucketWrapper returns an implementation of the gcsObjectIterator interface.
func (w *gcsBucketWrapper) objects(ctx context.Context, q *storage.Query) gcsObjectIterator {
	// This is the real gcs iterator.
	realIterator := w.bucket.Objects(ctx, q)
	// We return a wrapper around the real iterator.
	return &gcsObjectIteratorWrapper{iter: realIterator}
}

// gcsObjectWrapper wraps a storage.ObjectHandle to satisfy the gcsObject interface.
type gcsObjectWrapper struct {
	object *storage.ObjectHandle
}

// NewWriter implements the gcsObject interface for gcsObjectWrapper.
func (w *gcsObjectWrapper) newWriter(ctx context.Context) gcsWriter {
	return &gcsWriterWrapper{w: w.object.NewWriter(ctx)}
}

// NewReader implements the gcsObject interface for gcsObjectWrapper.
func (w *gcsObjectWrapper) newReader(ctx context.Context) (io.ReadCloser, error) {
	return w.object.NewReader(ctx)
}

// Delete implements the gcsObject interface for gcsObjectWrapper.
func (w *gcsObjectWrapper) delete(ctx context.Context) error {
	return w.object.Delete(ctx)
}

// Attrs implements the gcsObject interface for gcsObjectWrapper.
func (w *gcsObjectWrapper) attrs(ctx context.Context) (*storage.ObjectAttrs, error) {
	return w.object.Attrs(ctx)
}

// Create the wrapper for the real iterator.
type gcsObjectIteratorWrapper struct {
	iter *storage.ObjectIterator
}

func (w *gcsObjectIteratorWrapper) next() (*storage.ObjectAttrs, error) {
	return w.iter.Next()
}

// gcsWriterWrapper wraps the real gcs writer to satisfy our ObjectWriter interface.
type gcsWriterWrapper struct {
	w *storage.Writer
}

func (g *gcsWriterWrapper) Write(p []byte) (n int, err error) {
	return g.w.Write(p)
}

func (g *gcsWriterWrapper) Close() error {
	return g.w.Close()
}

func (g *gcsWriterWrapper) SetContentType(cType string) {
	g.w.ContentType = cType
}

var _ gcsClient = (*gcsClientWrapper)(nil)
var _ gcsBucket = (*gcsBucketWrapper)(nil)
var _ gcsObject = (*gcsObjectWrapper)(nil)
var _ gcsObjectIterator = (*gcsObjectIteratorWrapper)(nil)
var _ gcsWriter = (*gcsWriterWrapper)(nil)

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

var _ gcsClient = (*fakeClient)(nil)
var _ gcsBucket = (*fakeBucket)(nil)
var _ gcsObject = (*fakeObject)(nil)
var _ gcsObjectIterator = (*fakeObjectIterator)(nil)
var _ gcsWriter = (*fakeWriter)(nil)
