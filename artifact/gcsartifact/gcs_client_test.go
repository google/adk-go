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
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"cloud.google.com/go/storage"
	"google.golang.org/api/option"
)

// TestObjectWrapperIfNotExist checks that the wrapper puts the does-not-exist
// precondition on the wire. Everything else in this package tests against the
// fake, so without this a wrapper that dropped the condition would still pass
// while Save silently degraded to last-writer-wins. The client sends it as the
// ifGenerationMatch=0 query parameter on the upload URL.
func TestObjectWrapperIfNotExist(t *testing.T) {
	for _, tc := range []struct {
		name        string
		conditional bool
		want        string
	}{
		{"conditional", true, "0"},
		{"unconditional", false, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var got string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				got = r.URL.Query().Get("ifGenerationMatch")
				if _, err := io.Copy(io.Discard, r.Body); err != nil {
					t.Errorf("draining request body failed: %v", err)
				}
				w.Header().Set("Content-Type", "application/json")
				if _, err := io.WriteString(w, `{"bucket":"bucket","name":"obj","generation":"1"}`); err != nil {
					t.Errorf("writing response failed: %v", err)
				}
			}))
			defer srv.Close()

			client, err := storage.NewClient(t.Context(), option.WithEndpoint(srv.URL), option.WithoutAuthentication())
			if err != nil {
				t.Fatalf("storage.NewClient() failed: %v", err)
			}
			t.Cleanup(func() {
				if err := client.Close(); err != nil {
					t.Errorf("client.Close() failed: %v", err)
				}
			})

			obj := (&gcsClientWrapper{client: client}).bucket("bucket").object("obj")
			if tc.conditional {
				obj = obj.ifNotExist()
			}
			w := obj.newWriter(t.Context())
			if _, err := w.Write([]byte("data")); err != nil {
				t.Fatalf("Write() failed: %v", err)
			}
			if err := w.Close(); err != nil {
				t.Fatalf("Close() failed: %v", err)
			}

			if got != tc.want {
				t.Errorf("ifGenerationMatch = %q, want %q", got, tc.want)
			}
		})
	}
}
