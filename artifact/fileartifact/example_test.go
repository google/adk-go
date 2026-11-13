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

package fileartifact_test

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/artifact/fileartifact"
)

// The service keeps each version in its own directory, with the payload and a
// metadata.json beside it. A "user:" file name is stored in the user's
// directory, outside every session, and List reports it in each session.
func ExampleNewService() {
	root, err := os.MkdirTemp("", "artifacts")
	if err != nil {
		log.Fatal(err)
	}
	defer removeAll(root)

	ctx := context.Background()
	svc, err := fileartifact.NewService(ctx, root)
	if err != nil {
		log.Fatal(err)
	}

	for _, req := range []*artifact.SaveRequest{
		{FileName: "report.txt", Part: genai.NewPartFromText("quarterly numbers")},
		{FileName: "user:avatar.png", Part: genai.NewPartFromBytes([]byte{0x89, 'P', 'N', 'G'}, "image/png")},
	} {
		req.AppName, req.UserID, req.SessionID = "app", "alice", "s1"
		if _, err := svc.Save(ctx, req); err != nil {
			log.Fatal(err)
		}
	}

	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, err := filepath.Rel(root, path)
		fmt.Println(filepath.ToSlash(rel))
		return err
	})
	if err != nil {
		log.Fatal(err)
	}

	list, err := svc.List(ctx, &artifact.ListRequest{AppName: "app", UserID: "alice", SessionID: "s2"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("session s2 lists %q\n", list.FileNames)

	loaded, err := svc.Load(ctx, &artifact.LoadRequest{AppName: "app", UserID: "alice", SessionID: "s1", FileName: "report.txt"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("report.txt loads", loaded.Part.Text)
	// Output:
	// apps/app/users/alice/artifacts/avatar.png/versions/1/avatar.png
	// apps/app/users/alice/artifacts/avatar.png/versions/1/metadata.json
	// apps/app/users/alice/sessions/s1/artifacts/report.txt/versions/1/metadata.json
	// apps/app/users/alice/sessions/s1/artifacts/report.txt/versions/1/report.txt
	// session s2 lists ["user:avatar.png"]
	// report.txt loads quarterly numbers
}

// The service refuses file data, which it cannot store, and names that hold a
// path separator. Both errors wrap artifact.ErrInputValidation.
func ExampleNewService_refused() {
	root, err := os.MkdirTemp("", "artifacts")
	if err != nil {
		log.Fatal(err)
	}
	defer removeAll(root)

	ctx := context.Background()
	svc, err := fileartifact.NewService(ctx, root)
	if err != nil {
		log.Fatal(err)
	}

	for _, req := range []*artifact.SaveRequest{
		{FileName: "remote.pdf", Part: genai.NewPartFromURI("gs://bucket/remote.pdf", "application/pdf")},
		{FileName: "docs/report.txt", Part: genai.NewPartFromText("nested")},
	} {
		req.AppName, req.UserID, req.SessionID = "app", "alice", "s1"
		_, err := svc.Save(ctx, req)
		fmt.Printf("%s: invalid input: %t\n", req.FileName, errors.Is(err, artifact.ErrInputValidation))
	}
	// Output:
	// remote.pdf: invalid input: true
	// docs/report.txt: invalid input: true
}

// removeAll deletes the example's directory tree.
func removeAll(dir string) {
	if err := os.RemoveAll(dir); err != nil {
		log.Print(err)
	}
}
