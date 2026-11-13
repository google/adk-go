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
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"cloud.google.com/go/storage"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/option"
	"google.golang.org/genai"
)

func TestGCSWriterWrapperSetMetadata(t *testing.T) {
	writer := &storage.Writer{}
	wrapper := &gcsWriterWrapper{w: writer}
	want := map[string]string{"key": "value"}

	wrapper.SetMetadata(want)

	if diff := cmp.Diff(want, writer.Metadata); diff != "" {
		t.Errorf("writer.Metadata mismatch (-want +got):\n%s", diff)
	}
}

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

// TestWriterWrapperDoesNotSniffEmptyContentType checks what the upload carries
// as the object's content type. The storage client sniffs a blank content type
// from the payload, and a zero-byte file-data blob would then read back as
// "text/plain; charset=utf-8" instead of no MIME type.
func TestWriterWrapperDoesNotSniffEmptyContentType(t *testing.T) {
	for _, tc := range []struct {
		name        string
		contentType string
		wantInBody  string
		wantMissing string
	}{
		{name: "empty", contentType: "", wantMissing: "text/plain"},
		{name: "set", contentType: "application/pdf", wantInBody: "application/pdf"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var body []byte
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var err error
				if body, err = io.ReadAll(r.Body); err != nil {
					t.Errorf("reading request body failed: %v", err)
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

			w := (&gcsClientWrapper{client: client}).bucket("bucket").object("obj").newWriter(t.Context())
			w.SetMetadata(map[string]string{"adkFileUri": "gs://b/o"})
			w.SetContentType(tc.contentType)
			if err := w.Close(); err != nil {
				t.Fatalf("Close() failed: %v", err)
			}

			if tc.wantInBody != "" && !strings.Contains(string(body), tc.wantInBody) {
				t.Errorf("upload does not carry content type %q", tc.wantInBody)
			}
			if tc.wantMissing != "" && strings.Contains(string(body), tc.wantMissing) {
				t.Errorf("upload carries a sniffed content type %q", tc.wantMissing)
			}
		})
	}
}

// signingBucket is the fake bucket with the signing of a real bucket handle.
type signingBucket struct {
	*fakeBucket
	signer gcsBucket
}

func (b *signingBucket) signedURL(object string, opts *storage.SignedURLOptions) (string, error) {
	return b.signer.signedURL(object, opts)
}

// TestDefaultSignerUsesClientCredentials checks that GetSignedURL without a
// Signer signs with the credentials of the client NewService built, offline,
// from a service account key.
func TestDefaultSignerUsesClientCredentials(t *testing.T) {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("rsa.GenerateKey() failed: %v", err)
	}
	der, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatalf("x509.MarshalPKCS8PrivateKey() failed: %v", err)
	}
	creds, err := json.Marshal(map[string]string{
		"type":           "service_account",
		"project_id":     "p",
		"private_key_id": "k",
		"private_key":    string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})),
		"client_email":   "s@p.iam.gserviceaccount.com",
		"client_id":      "1",
		"token_uri":      "https://oauth2.googleapis.com/token",
	})
	if err != nil {
		t.Fatalf("json.Marshal() failed: %v", err)
	}
	client, err := storage.NewClient(t.Context(), option.WithAuthCredentialsJSON(option.ServiceAccount, creds))
	if err != nil {
		t.Fatalf("storage.NewClient() failed: %v", err)
	}
	t.Cleanup(func() {
		if err := client.Close(); err != nil {
			t.Errorf("client.Close() failed: %v", err)
		}
	})
	svc := newGCSServiceForTesting("b")
	svc.bucket = &signingBucket{fakeBucket: svc.bucket.(*fakeBucket), signer: (&gcsClientWrapper{client: client}).bucket("b")}
	mustSave(t, svc, "sess1", "notes.txt", genai.NewPartFromText("n"))

	got, err := svc.GetSignedURL(t.Context(), &GetSignedURLRequest{
		AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "notes.txt",
	})
	if err != nil {
		t.Fatalf("GetSignedURL() failed: %v", err)
	}
	if want := "https://storage.googleapis.com/b/app/user1/sess1/notes.txt/1?Expires="; !strings.HasPrefix(got, want) {
		t.Errorf("GetSignedURL() = %q, want prefix %q", got, want)
	}
	if want := "GoogleAccessId=s%40p.iam.gserviceaccount.com"; !strings.Contains(got, want) {
		t.Errorf("GetSignedURL() = %q, want it to contain %q", got, want)
	}

	// The client escapes the object name itself, so the service must hand it
	// the raw name: an escaped one would come back escaped twice.
	mustSave(t, svc, "sess1", "user:profile.png", genai.NewPartFromText("p"))
	got, err = svc.GetSignedURL(t.Context(), &GetSignedURLRequest{
		AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "user:profile.png",
	})
	if err != nil {
		t.Fatalf("GetSignedURL(user:profile.png) failed: %v", err)
	}
	if want := "https://storage.googleapis.com/b/app/user1/user/user:profile.png/1?Expires="; !strings.HasPrefix(got, want) {
		t.Errorf("GetSignedURL(user:profile.png) = %q, want prefix %q", got, want)
	}
}
