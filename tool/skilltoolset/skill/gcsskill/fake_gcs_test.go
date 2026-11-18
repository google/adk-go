// Copyright 2026 Google LLC
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

package gcsskill

import (
	"encoding/json"
	"errors"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"cloud.google.com/go/storage"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"
)

// fakeGCS stands in for Cloud Storage in tests. The storage client has no
// mock, so the tests give a real [storage.Client] this HTTP server instead of
// the service. It answers the two JSON API calls the source makes: listing
// the objects of a bucket and reading one object. A test therefore runs the
// source through [NewSource], the way an application does.
type fakeGCS struct {
	// client is a real storage client that talks to this fake.
	client *storage.Client

	// mu guards the fields below it, which the methods after ServeHTTP set.
	mu             sync.Mutex
	objects        map[string][]byte
	pageSize       int
	failedListPage int
	directoryPages [][]string
	readStatus     map[string]int
	cutRead        map[string]int

	// listRequests and readRequests count the requests received.
	listRequests atomic.Int64
	readRequests atomic.Int64
	// openBodies counts the HTTP responses whose body the client has not
	// closed. A source that forgets to close an object leaves one open.
	openBodies atomic.Int64
}

// testBucket is the name of the bucket that a fakeGCS holds.
const testBucket = "test-bucket"

// mustNewFakeGCS starts a fake Cloud Storage that holds objects in testBucket,
// or fails the test. A test passes its client to [NewSource].
func mustNewFakeGCS(t *testing.T, objects map[string][]byte) *fakeGCS {
	t.Helper()
	fake := &fakeGCS{
		objects:    maps.Clone(objects),
		readStatus: make(map[string]int),
		cutRead:    make(map[string]int),
	}
	server := httptest.NewServer(fake)
	t.Cleanup(server.Close)

	client, err := storage.NewClient(
		t.Context(),
		option.WithEndpoint(server.URL+"/storage/v1/"),
		option.WithHTTPClient(&http.Client{Transport: &bodyCounter{base: server.Client().Transport, open: &fake.openBodies}}),
		storage.WithJSONReads(),
	)
	if err != nil {
		t.Fatalf("storage.NewClient() failed: %v", err)
	}
	fake.client = client
	t.Cleanup(func() {
		if open := fake.openBodies.Load(); open != 0 {
			t.Errorf("%d response bodies were left open", open)
		}
		if err := client.Close(); err != nil {
			t.Errorf("client.Close() failed: %v", err)
		}
	})
	return fake
}

// failedWith reports whether err is the error of a Cloud Storage request that
// failed with the HTTP status.
func failedWith(err error, status int) bool {
	var apiErr *googleapi.Error
	return errors.As(err, &apiErr) && apiErr.Code == status
}

// ServeHTTP serves "/storage/v1/b/test-bucket/o", the listing, and
// "/storage/v1/b/test-bucket/o/<object>", one object. The client escapes the
// object name, slashes included, into one path segment.
func (f *fakeGCS) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	rest, ok := strings.CutPrefix(r.URL.EscapedPath(), "/storage/v1/b/"+testBucket+"/o")
	if !ok {
		writeError(w, http.StatusNotFound)
		return
	}
	if rest == "" {
		f.listRequests.Add(1)
		f.serveList(w, r)
		return
	}
	name, err := url.PathUnescape(strings.TrimPrefix(rest, "/"))
	if err != nil {
		writeError(w, http.StatusBadRequest)
		return
	}
	f.readRequests.Add(1)
	f.serveObject(w, name)
}

func (f *fakeGCS) serveObject(w http.ResponseWriter, name string) {
	f.mu.Lock()
	data, ok := f.objects[name]
	status := f.readStatus[name]
	cut, hasCut := f.cutRead[name]
	if hasCut {
		delete(f.cutRead, name)
		f.readStatus[name] = http.StatusForbidden
	}
	f.mu.Unlock()

	switch {
	case status != 0:
		writeError(w, status)
	case !ok:
		writeError(w, http.StatusNotFound)
	case hasCut:
		w.Header().Set("Content-Length", strconv.Itoa(len(data)))
		_, _ = w.Write(data[:cut])
		w.(http.Flusher).Flush()
		panic(http.ErrAbortHandler) // Drops the connection without the rest.
	default:
		w.Header().Set("Content-Length", strconv.Itoa(len(data)))
		_, _ = w.Write(data)
	}
}

func (f *fakeGCS) serveList(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	prefix, delimiter := query.Get("prefix"), query.Get("delimiter")
	// The page token is the index of the page.
	index, _ := strconv.Atoi(query.Get("pageToken"))

	f.mu.Lock()
	defer f.mu.Unlock()
	if index+1 == f.failedListPage {
		writeError(w, http.StatusForbidden)
		return
	}

	var pages [][]listEntry
	if delimiter != "" && f.directoryPages != nil {
		for _, directories := range f.directoryPages {
			var page []listEntry
			for _, directory := range directories {
				page = append(page, listEntry{directory: directory})
			}
			pages = append(pages, page)
		}
	} else {
		// Entries are the objects under the prefix in name order. With a
		// delimiter, the objects of one directory fold into its name.
		var names []string
		for name := range f.objects {
			if strings.HasPrefix(name, prefix) {
				names = append(names, name)
			}
		}
		slices.Sort(names)
		var entries []listEntry
		for _, name := range names {
			if end := strings.Index(name[len(prefix):], delimiter); delimiter != "" && end >= 0 {
				directory := name[:len(prefix)+end+len(delimiter)]
				if len(entries) == 0 || entries[len(entries)-1].directory != directory {
					entries = append(entries, listEntry{directory: directory})
				}
				continue
			}
			entries = append(entries, listEntry{object: name})
		}
		size := f.pageSize
		if size <= 0 {
			size = max(len(entries), 1)
		}
		pages = slices.Collect(slices.Chunk(entries, size))
	}

	response := map[string]any{"kind": "storage#objects"}
	if index < len(pages) {
		var objects []map[string]string
		var directories []string
		for _, entry := range pages[index] {
			if entry.directory != "" {
				directories = append(directories, entry.directory)
			} else {
				objects = append(objects, map[string]string{"name": entry.object})
			}
		}
		if objects != nil {
			response["items"] = objects
		}
		if directories != nil {
			response["prefixes"] = directories
		}
	}
	if index+1 < len(pages) {
		response["nextPageToken"] = strconv.Itoa(index + 1)
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(response)
}

// removeObjects deletes the objects whose names start with prefix.
func (f *fakeGCS) removeObjects(prefix string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	maps.DeleteFunc(f.objects, func(name string, _ []byte) bool {
		return strings.HasPrefix(name, prefix)
	})
}

// setPageSize puts size entries on each list page. By default a listing is one
// page.
func (f *fakeGCS) setPageSize(size int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.pageSize = size
}

// failListPage makes the list page of that number, counted from 1, fail with
// 403 Forbidden.
func (f *fakeGCS) failListPage(page int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.failedListPage = page
}

// setDirectoryPages makes a directory listing return these pages in place of
// the directories that the objects make up. It lets a test list directories
// out of order, or one directory on two pages.
func (f *fakeGCS) setDirectoryPages(pages [][]string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.directoryPages = pages
}

// failRead makes the read of the object fail with the HTTP status.
func (f *fakeGCS) failRead(object string, status int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.readStatus[object] = status
}

// cutReadAfter makes the read of the object lose its connection after size
// bytes. The client then asks for the rest, which fails with 403 Forbidden.
func (f *fakeGCS) cutReadAfter(object string, size int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.cutRead[object] = size
}

// listEntry is one entry of a listing: an object, or a directory that stands
// for the objects below it.
type listEntry struct {
	object    string
	directory string
}

func writeError(w http.ResponseWriter, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"code": status, "message": http.StatusText(status)},
	})
}

// bodyCounter is an [http.RoundTripper] that counts the response bodies that
// are open.
type bodyCounter struct {
	base http.RoundTripper
	open *atomic.Int64
}

func (c *bodyCounter) RoundTrip(r *http.Request) (*http.Response, error) {
	response, err := c.base.RoundTrip(r)
	if err != nil {
		return nil, err
	}
	c.open.Add(1)
	response.Body = &countedBody{ReadCloser: response.Body, open: c.open}
	return response, nil
}

type countedBody struct {
	io.ReadCloser
	open   *atomic.Int64
	closed atomic.Bool
}

func (b *countedBody) Close() error {
	if b.closed.CompareAndSwap(false, true) {
		b.open.Add(-1)
	}
	return b.ReadCloser.Close()
}
