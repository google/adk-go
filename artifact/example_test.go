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

package artifact_test

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
)

// Every save of a file name creates a new version, numbered from 1. A load
// with Version 0 returns the latest version; any other version is loaded as
// given, and a version that does not exist is an error wrapping
// fs.ErrNotExist.
func ExampleInMemoryService() {
	ctx := context.Background()
	svc := artifact.InMemoryService()

	for _, text := range []string{"draft", "final"} {
		resp, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "alice", SessionID: "s1",
			FileName: "report.txt",
			Part:     genai.NewPartFromText(text),
		})
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("saved version", resp.Version)
	}

	versions, err := svc.Versions(ctx, &artifact.VersionsRequest{
		AppName: "app", UserID: "alice", SessionID: "s1", FileName: "report.txt",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("versions:", versions.Versions)

	for _, version := range []int64{0, 1} {
		loaded, err := svc.Load(ctx, &artifact.LoadRequest{
			AppName: "app", UserID: "alice", SessionID: "s1", FileName: "report.txt",
			Version: version,
		})
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("version %d loads %q\n", version, loaded.Part.Text)
	}

	_, err = svc.Load(ctx, &artifact.LoadRequest{
		AppName: "app", UserID: "alice", SessionID: "s1", FileName: "report.txt",
		Version: 3,
	})
	fmt.Println("version 3 not found:", errors.Is(err, fs.ErrNotExist))
	// Output:
	// saved version 1
	// saved version 2
	// versions: [1 2]
	// version 0 loads "final"
	// version 1 loads "draft"
	// version 3 not found: true
}

// A file name with the "user:" prefix names an artifact of the user, shared by
// all the user's sessions. Requests for it need no session ID, and List
// reports it in every session of the user. List without a session ID returns
// only the user's "user:" file names.
func ExampleInMemoryService_userScope() {
	ctx := context.Background()
	svc := artifact.InMemoryService()

	save := func(sessionID, fileName, text string) {
		_, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "alice", SessionID: sessionID,
			FileName: fileName,
			Part:     genai.NewPartFromText(text),
		})
		if err != nil {
			log.Fatal(err)
		}
	}
	save("s1", "notes.txt", "seen only in s1")
	save("s1", "user:settings.json", `{"theme": "dark"}`)

	for _, sessionID := range []string{"s1", "s2", ""} {
		list, err := svc.List(ctx, &artifact.ListRequest{AppName: "app", UserID: "alice", SessionID: sessionID})
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("session %q lists %q\n", sessionID, list.FileNames)
	}

	loaded, err := svc.Load(ctx, &artifact.LoadRequest{AppName: "app", UserID: "alice", FileName: "user:settings.json"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("loaded without a session:", loaded.Part.Text)
	// Output:
	// session "s1" lists ["notes.txt" "user:settings.json"]
	// session "s2" lists ["user:settings.json"]
	// session "" lists ["user:settings.json"]
	// loaded without a session: {"theme": "dark"}
}

// An artifact can hold a reference to another artifact of the same user: a
// file-data part whose URI is the target's artifact:// URI. Load follows the
// reference and returns the target's content. A reference that leaves the
// caller's app, user or session is refused with an error wrapping
// ErrInputValidation.
func ExampleInMemoryService_reference() {
	ctx := context.Background()
	svc := artifact.InMemoryService()

	saved, err := svc.Save(ctx, &artifact.SaveRequest{
		AppName: "app", UserID: "alice", SessionID: "s1",
		FileName: "chart.png",
		Part:     genai.NewPartFromBytes([]byte{0x89, 'P', 'N', 'G'}, "image/png"),
	})
	if err != nil {
		log.Fatal(err)
	}

	ref := artifact.BuildArtifactURI("app", "alice", "s1", "chart.png", saved.Version)
	if _, err := svc.Save(ctx, &artifact.SaveRequest{
		AppName: "app", UserID: "alice", SessionID: "s1",
		FileName: "latest_chart",
		Part:     genai.NewPartFromURI(ref, "image/png"),
	}); err != nil {
		log.Fatal(err)
	}
	loaded, err := svc.Load(ctx, &artifact.LoadRequest{
		AppName: "app", UserID: "alice", SessionID: "s1", FileName: "latest_chart",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("latest_chart loads %s, %d bytes\n", loaded.Part.InlineData.MIMEType, len(loaded.Part.InlineData.Data))

	otherUser := artifact.BuildArtifactURI("app", "bob", "s1", "chart.png", 1)
	_, err = svc.Save(ctx, &artifact.SaveRequest{
		AppName: "app", UserID: "alice", SessionID: "s1",
		FileName: "bobs_chart",
		Part:     genai.NewPartFromURI(otherUser, "image/png"),
	})
	fmt.Println(err)
	fmt.Println("invalid input:", errors.Is(err, artifact.ErrInputValidation))
	// Output:
	// latest_chart loads image/png, 4 bytes
	// request validation failed: invalid artifact reference: artifact reference must stay within the same app and user scope
	// invalid input: true
}

// GetArtifactVersion returns the record of one version, or of the latest one
// for Version 0. Custom metadata values come back as strings.
func ExampleService_GetArtifactVersion() {
	ctx := context.Background()
	svc := artifact.InMemoryService()

	if _, err := svc.Save(ctx, &artifact.SaveRequest{
		AppName: "app", UserID: "alice", SessionID: "s1",
		FileName:       "chart.png",
		Part:           genai.NewPartFromBytes([]byte{0x89, 'P', 'N', 'G'}, "image/png"),
		CustomMetadata: map[string]any{"pages": 3, "draft": true},
	}); err != nil {
		log.Fatal(err)
	}

	resp, err := svc.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{
		AppName: "app", UserID: "alice", SessionID: "s1", FileName: "chart.png",
	})
	if err != nil {
		log.Fatal(err)
	}
	v := resp.ArtifactVersion
	fmt.Println("version:", v.Version)
	fmt.Println("media type:", v.MimeType)
	fmt.Println("canonical URI:", v.CanonicalURI)
	fmt.Printf("pages: %q, draft: %q\n", v.CustomMetadata["pages"], v.CustomMetadata["draft"])
	// Output:
	// version: 1
	// media type: image/png
	// canonical URI: memory://apps/app/users/alice/sessions/s1/artifacts/chart.png/versions/1
	// pages: "3", draft: "true"
}

func ExampleBuildArtifactURI() {
	fmt.Println(artifact.BuildArtifactURI("app", "alice", "s1", "report.pdf", 2))
	// An empty session ID builds the URI of a user-scoped artifact.
	fmt.Println(artifact.BuildArtifactURI("app", "alice", "", "user:avatar.png", 1))
	// Output:
	// artifact://apps/app/users/alice/sessions/s1/artifacts/report.pdf/versions/2
	// artifact://apps/app/users/alice/artifacts/user:avatar.png/versions/1
}

// ParseArtifactURI accepts app, user and session IDs that contain "/", such as
// a Vertex AI Agent Engine resource name, as long as they do not contain one
// of the words that separate the segments of the URI between two slashes.
func ExampleParseArtifactURI() {
	uri := "artifact://apps/projects/123/locations/us-central1/reasoningEngines/456" +
		"/users/alice/sessions/s1/artifacts/report.pdf/versions/2"
	parsed, ok := artifact.ParseArtifactURI(uri)
	fmt.Println(ok)
	fmt.Println("app:", parsed.AppName)
	fmt.Println("user:", parsed.UserID)
	fmt.Println("session:", parsed.SessionID)
	fmt.Println("file:", parsed.FileName)
	fmt.Println("version:", parsed.Version)

	_, ok = artifact.ParseArtifactURI("gs://bucket/report.pdf")
	fmt.Println(ok)
	// Output:
	// true
	// app: projects/123/locations/us-central1/reasoningEngines/456
	// user: alice
	// session: s1
	// file: report.pdf
	// version: 2
	// false
}

func ExampleValidatePathSegment() {
	for _, id := range []string{"alice", "team/alice", "../escape", "C:temp", "team/users/alice"} {
		fmt.Printf("%q: %v\n", id, artifact.ValidatePathSegment(id, "user_id"))
	}
	// Output:
	// "alice": <nil>
	// "team/alice": <nil>
	// "../escape": user_id must not contain traversal segments.
	// "C:temp": user_id must not be drive-qualified.
	// "team/users/alice": user_id must not contain reserved path segments.
}

// The services check every request before they touch storage. errors.Is with
// ErrInputValidation tells a request the caller must fix from an artifact
// that does not exist.
func ExampleErrInputValidation() {
	ctx := context.Background()
	svc := artifact.InMemoryService()

	for _, sessionID := range []string{"../s2", "s1"} {
		_, err := svc.Load(ctx, &artifact.LoadRequest{
			AppName: "app", UserID: "alice", SessionID: sessionID, FileName: "report.txt",
		})
		switch {
		case errors.Is(err, artifact.ErrInputValidation):
			fmt.Printf("session %q: invalid input: %v\n", sessionID, err)
		case errors.Is(err, fs.ErrNotExist):
			fmt.Printf("session %q: not found\n", sessionID)
		}
	}
	// Output:
	// session "../s2": invalid input: request validation failed: session_id must not contain traversal segments.
	// session "s1": not found
}

// A session rewind removes an artifact by saving the rewind marker as its
// newest version. The in-memory service then loads the artifact as not found,
// while the versions saved before the marker still load.
func ExampleNewRewindTombstone() {
	ctx := context.Background()
	svc := artifact.InMemoryService()

	for _, part := range []*genai.Part{genai.NewPartFromText("draft"), artifact.NewRewindTombstone()} {
		if _, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName: "app", UserID: "alice", SessionID: "s1", FileName: "report.txt", Part: part,
		}); err != nil {
			log.Fatal(err)
		}
	}

	_, err := svc.Load(ctx, &artifact.LoadRequest{
		AppName: "app", UserID: "alice", SessionID: "s1", FileName: "report.txt",
	})
	fmt.Println("latest not found:", errors.Is(err, fs.ErrNotExist))

	first, err := svc.Load(ctx, &artifact.LoadRequest{
		AppName: "app", UserID: "alice", SessionID: "s1", FileName: "report.txt", Version: 1,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("version 1:", first.Part.Text)
	fmt.Println("marker:", artifact.IsRewindTombstone(artifact.NewRewindTombstone()))
	// Output:
	// latest not found: true
	// version 1: draft
	// marker: true
}
