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

package gcsartifact_test

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
	"time"

	"cloud.google.com/go/storage"
	"google.golang.org/api/option"

	"google.golang.org/adk/v2/artifact/gcsartifact"
)

// newFakeStorage starts a local server for the Cloud Storage JSON API that
// knows a single object, so the example runs without a bucket or network.
func newFakeStorage(bucket, object string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/storage/v1/b/"+bucket+"/o/"+object {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := fmt.Fprintf(w, `{"bucket": %q, "name": %q, "contentType": "application/pdf"}`, bucket, object); err != nil {
			log.Printf("fake storage: writing the response: %v", err)
		}
	}))
}

func ExampleURLProvider() {
	fake := newFakeStorage("my-bucket", "app/user1/session1/report.pdf/1")
	defer fake.Close()

	ctx := context.Background()
	// Without the two options, NewService connects to Cloud Storage with
	// Application Default Credentials.
	svc, err := gcsartifact.NewService(ctx, "my-bucket",
		option.WithEndpoint(fake.URL+"/storage/v1/"), option.WithoutAuthentication())
	if err != nil {
		log.Fatal(err)
	}
	urls := svc.(gcsartifact.URLProvider)

	version := int64(1)
	url, err := urls.GetSignedURL(ctx, &gcsartifact.GetSignedURLRequest{
		AppName:    "app",
		UserID:     "user1",
		SessionID:  "session1",
		FileName:   "report.pdf",
		Version:    &version,
		Expiration: 15 * time.Minute,
		// A signer that calls your key service; this one only formats the URL.
		Signer: func(bucket, object string, opts *storage.SignedURLOptions) (string, error) {
			return fmt.Sprintf("https://storage.googleapis.com/%s/%s?method=%s", bucket, object, opts.Method), nil
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(url)
	// Output:
	// https://storage.googleapis.com/my-bucket/app/user1/session1/report.pdf/1?method=GET
}
