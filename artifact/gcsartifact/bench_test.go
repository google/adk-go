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

package gcsartifact

import (
	"context"
	"fmt"
	"io"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"cloud.google.com/go/storage"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
)

// rpcDelay stands in for one round trip to GCS. Every call that is an RPC on
// the real client (list, metadata, open for read, upload, delete) sleeps this
// long, so a round trip saved shows up in wall-clock time too.
const rpcDelay = 20 * time.Millisecond

// rpcCounter wraps the in-memory fake and counts the calls that are RPCs on
// the real client.
type rpcCounter struct {
	rpcs atomic.Int64
	// unprojectedLists counts object listings issued without an attribute
	// selection, which make GCS return every attribute of every object.
	unprojectedLists atomic.Int64
	delay            time.Duration
}

func (c *rpcCounter) rpc() {
	c.rpcs.Add(1)
	if c.delay > 0 {
		time.Sleep(c.delay)
	}
}

func (c *rpcCounter) reset() {
	c.rpcs.Store(0)
	c.unprojectedLists.Store(0)
}

type countingBucket struct {
	gcsBucket
	c *rpcCounter
}

func (b *countingBucket) object(name string) gcsObject {
	return &countingObject{gcsObject: b.gcsBucket.object(name), c: b.c}
}

func (b *countingBucket) objects(ctx context.Context, q *storage.Query) gcsObjectIterator {
	b.c.rpc()
	// storage.Query keeps the selection private; read its length to tell a
	// projected listing from a full one.
	if q == nil || reflect.ValueOf(q).Elem().FieldByName("attrSelection").Len() == 0 {
		b.c.unprojectedLists.Add(1)
	}
	return b.gcsBucket.objects(ctx, q)
}

type countingObject struct {
	gcsObject
	c *rpcCounter
}

func (o *countingObject) newWriter(ctx context.Context) gcsWriter {
	return &countingWriter{gcsWriter: o.gcsObject.newWriter(ctx), c: o.c}
}

func (o *countingObject) newReader(ctx context.Context) (io.ReadCloser, error) {
	o.c.rpc()
	return o.gcsObject.newReader(ctx)
}

func (o *countingObject) delete(ctx context.Context) error {
	o.c.rpc()
	return o.gcsObject.delete(ctx)
}

func (o *countingObject) attrs(ctx context.Context) (*storage.ObjectAttrs, error) {
	o.c.rpc()
	return o.gcsObject.attrs(ctx)
}

func (o *countingObject) ifNotExist() gcsObject {
	return &countingObject{gcsObject: o.gcsObject.ifNotExist(), c: o.c}
}

type countingWriter struct {
	gcsWriter
	c *rpcCounter
}

// Close is the upload: the real writer sends the object when it is closed.
func (w *countingWriter) Close() error {
	w.c.rpc()
	return w.gcsWriter.Close()
}

func benchService(b *testing.B) (*gcsService, *rpcCounter) {
	b.Helper()
	c := &rpcCounter{delay: rpcDelay}
	client := newFakeClient()
	bucket := &countingBucket{gcsBucket: client.bucket("bucket"), c: c}
	return &gcsService{
		bucketName:    "bucket",
		storageClient: client,
		bucket:        bucket,
		sleep:         func(ctx context.Context, _ time.Duration) error { return ctx.Err() },
	}, c
}

func (c *rpcCounter) report(b *testing.B) {
	b.Helper()
	b.ReportMetric(float64(c.rpcs.Load())/float64(b.N), "rpcs/op")
	b.ReportMetric(float64(c.unprojectedLists.Load())/float64(b.N), "unprojected-lists/op")
}

// BenchmarkGCSLoad loads a 4 KiB artifact, by latest version (a listing
// resolves it first) and by explicit version.
func BenchmarkGCSLoad(b *testing.B) {
	for _, tc := range []struct {
		name    string
		version int64
	}{
		{name: "latest", version: 0},
		{name: "explicit-version", version: 1},
	} {
		b.Run(tc.name, func(b *testing.B) {
			svc, c := benchService(b)
			req := saveReq()
			req.Part = genai.NewPartFromBytes(make([]byte, 4096), "application/octet-stream")
			if _, err := svc.Save(b.Context(), req); err != nil {
				b.Fatal(err)
			}
			load := &artifact.LoadRequest{
				AppName: req.AppName, UserID: req.UserID, SessionID: req.SessionID, FileName: req.FileName,
				Version: tc.version,
			}
			b.ReportAllocs()
			c.reset()
			for b.Loop() {
				resp, err := svc.Load(b.Context(), load)
				if err != nil {
					b.Fatal(err)
				}
				if resp.Part.InlineData.MIMEType != "application/octet-stream" {
					b.Fatalf("MIMEType = %q", resp.Part.InlineData.MIMEType)
				}
			}
			c.report(b)
		})
	}
}

// BenchmarkGCSSave saves a new artifact per iteration: one listing to pick
// the version, one upload.
func BenchmarkGCSSave(b *testing.B) {
	svc, c := benchService(b)
	b.ReportAllocs()
	c.reset()
	i := 0
	for b.Loop() {
		req := saveReq()
		req.FileName = fmt.Sprintf("file-%d", i)
		if _, err := svc.Save(b.Context(), req); err != nil {
			b.Fatal(err)
		}
		i++
	}
	c.report(b)
}
