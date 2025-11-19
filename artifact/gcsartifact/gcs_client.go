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
	"io"

	"cloud.google.com/go/storage"
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
	return &gcsBucketWrapper{
		bucket: w.client.Bucket(name),
	}
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

var (
	_ gcsClient         = (*gcsClientWrapper)(nil)
	_ gcsBucket         = (*gcsBucketWrapper)(nil)
	_ gcsObject         = (*gcsObjectWrapper)(nil)
	_ gcsObjectIterator = (*gcsObjectIteratorWrapper)(nil)
	_ gcsWriter         = (*gcsWriterWrapper)(nil)
)
