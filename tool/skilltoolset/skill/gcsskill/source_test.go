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
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"
	"testing/fstest"

	"cloud.google.com/go/storage"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/api/option"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
)

// Every test builds its source with the exported NewSource over the client of
// a fakeGCS (fake_gcs_test.go): a real storage client that talks to an
// in-process fake of Cloud Storage.

// pngHeader is valid binary content that is not valid UTF-8, matching
// _PNG_HEADER in adk-python's tests/unittests/skills/test__utils.py.
var pngHeader = []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}

func TestNewSource_Validation(t *testing.T) {
	// Creating a client issues no request.
	client, err := storage.NewClient(t.Context(), option.WithoutAuthentication())
	if err != nil {
		t.Fatalf("storage.NewClient() failed: %v", err)
	}
	t.Cleanup(func() {
		_ = client.Close() // Ignore error as the client was never used.
	})

	for _, tc := range []struct {
		name    string
		cfg     Config
		wantErr bool
	}{
		{name: "bucket root", cfg: Config{Client: client, Bucket: "b"}},
		{name: "base path", cfg: Config{Client: client, Bucket: "b", BasePath: "team-a/skills"}},
		{name: "nil client", cfg: Config{Bucket: "b"}, wantErr: true},
		{name: "empty bucket", cfg: Config{Client: client}, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewSource(tc.cfg)
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Fatalf("NewSource(%+v) error = %v, wantErr %t", tc.cfg, err, tc.wantErr)
			}
		})
	}
}

func TestGCSSource_ListFrontmatters(t *testing.T) {
	for _, tc := range []struct {
		name     string
		basePath string
		blobs    map[string][]byte
		want     []*skill.Frontmatter
		wantErr  error
		// wantErrText is what the error must say for the user to find the skill.
		wantErrText string
	}{
		{
			name: "root bucket ignores non-skills and deeper prefixes",
			blobs: map[string][]byte{
				"math-skill/SKILL.md":    []byte("---\nname: math-skill\ndescription: math\n---\n"),
				"weather-skill/SKILL.md": []byte("---\nname: weather-skill\ndescription: weather\n---\n"),
				"random-file.txt":        []byte("ignored"),
				"SKILL.md":               []byte("---\nname: root\ndescription: ignored\n---\n"),
				"dir/not-skill.txt":      []byte("ignored"),
				"sub/dir/SKILL.md":       []byte("---\nname: dir\ndescription: ignored\n---\n"),
			},
			want: []*skill.Frontmatter{
				{Name: "math-skill", Description: "math"},
				{Name: "weather-skill", Description: "weather"},
			},
		},
		{
			name:     "base path without trailing slash",
			basePath: "skills",
			blobs: map[string][]byte{
				"skills/my-skill/SKILL.md": []byte("---\nname: my-skill\ndescription: A skill\n---\nBody"),
				"other/ignored/SKILL.md":   []byte("---\nname: ignored\ndescription: other\n---\nBody"),
			},
			want: []*skill.Frontmatter{
				{Name: "my-skill", Description: "A skill"},
			},
		},
		{
			name:     "nested base path with slashes at both ends",
			basePath: "/org/team-a/skills/",
			blobs: map[string][]byte{
				"org/team-a/skills/my-skill/SKILL.md": []byte("---\nname: my-skill\ndescription: A skill\n---\nBody"),
				"org/team-a/skills/README.md":         []byte("ignored"),
				"org/team-a/skills/empty-dir/doc.txt": []byte("not a skill"),
			},
			want: []*skill.Frontmatter{
				{Name: "my-skill", Description: "A skill"},
			},
		},
		{
			name: "empty bucket",
		},
		{
			name: "allowed-tools scalar and list forms",
			blobs: map[string][]byte{
				"spec-skill/SKILL.md": []byte("---\nname: spec-skill\ndescription: test\nallowed-tools: Bash(git:*) Bash(jq:*) Read\n---\nInstructions."),
				"list-skill/SKILL.md": []byte("---\nname: list-skill\ndescription: test\nallowed-tools:\n  - Bash(git:*)\n  - Bash(jq:*)\n  - Read\n---\nInstructions."),
			},
			want: []*skill.Frontmatter{
				{Name: "list-skill", Description: "test", AllowedTools: []string{"Bash(git:*)", "Bash(jq:*)", "Read"}},
				{Name: "spec-skill", Description: "test", AllowedTools: []string{"Bash(git:*)", "Bash(jq:*)", "Read"}},
			},
		},
		{
			name:     "invalid frontmatter fails the listing",
			basePath: "skills/",
			blobs: map[string][]byte{
				"skills/invalid-skill/SKILL.md": []byte("invalid yaml content"),
				"skills/valid-skill/SKILL.md":   []byte("---\nname: valid-skill\ndescription: A skill\n---\nBody"),
			},
			wantErr:     skill.ErrInvalidFrontmatter,
			wantErrText: `"skills/invalid-skill/SKILL.md"`,
		},
		{
			name:     "name other than the directory fails the listing",
			basePath: "skills/",
			blobs: map[string][]byte{
				"skills/wrong-dir/SKILL.md": []byte("---\nname: my-skill\ndescription: A skill\n---\nBody"),
			},
			wantErr: skill.ErrInvalidSkillName,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fake := mustNewFakeGCS(t, tc.blobs)
			src, err := NewSource(Config{Client: fake.client, Bucket: testBucket, BasePath: tc.basePath})
			if err != nil {
				t.Fatalf("NewSource() failed: %v", err)
			}

			got, err := src.ListFrontmatters(t.Context())
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("ListFrontmatters() error = %v, want %v", err, tc.wantErr)
			}
			if tc.wantErr != nil {
				if !strings.Contains(err.Error(), tc.wantErrText) {
					t.Errorf("ListFrontmatters() error = %q, want it to contain %q", err, tc.wantErrText)
				}
				return
			}
			if diff := cmp.Diff(tc.want, got, cmpopts.SortSlices(func(a, b *skill.Frontmatter) bool { return a.Name < b.Name })); diff != "" {
				t.Errorf("ListFrontmatters() mismatch (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("object directly under the base path is not taken for a directory", func(t *testing.T) {
		fake := mustNewFakeGCS(t, map[string][]byte{
			"skills/README.md":      []byte("not a skill"),
			"skills/alpha/SKILL.md": []byte("---\nname: alpha\ndescription: Alpha\n---\nBody"),
		})
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket, BasePath: "skills"})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}

		got, err := src.ListFrontmatters(t.Context())
		if err != nil {
			t.Fatalf("ListFrontmatters() failed: %v", err)
		}
		if diff := cmp.Diff([]*skill.Frontmatter{{Name: "alpha", Description: "Alpha"}}, got); diff != "" {
			t.Errorf("ListFrontmatters() mismatch (-want +got):\n%s", diff)
		}
		if got := fake.readRequests.Load(); got != 1 {
			t.Errorf("read requests = %d, want 1 (the SKILL.md of alpha)", got)
		}
	})
}

func TestGCSSource_OnListInvalidSkill(t *testing.T) {
	type recordedFailure struct {
		name string
		err  error
	}

	objects := map[string][]byte{
		"skills/alpha-valid/SKILL.md":  []byte("---\nname: alpha-valid\ndescription: Valid alpha\n---\nBody"),
		"skills/bad-yaml/SKILL.md":     []byte("invalid yaml content"),
		"skills/mismatch/SKILL.md":     []byte("---\nname: wrong-name\ndescription: Mismatch\n---\nBody"),
		"skills/not-a-skill/notes.txt": []byte("directory without SKILL.md"),
		"skills/zeta-valid/SKILL.md":   []byte("---\nname: zeta-valid\ndescription: Valid zeta\n---\nBody"),
		// A directory named "..", which no skill can be named.
		"skills/../SKILL.md": []byte("---\nname: evil\ndescription: Evil\n---\nBody"),
	}

	t.Run("skips the skills for which the callback returns nil", func(t *testing.T) {
		var failures []recordedFailure
		src, err := NewSource(Config{
			Client:   mustNewFakeGCS(t, objects).client,
			Bucket:   testBucket,
			BasePath: "skills",
			OnListInvalidSkill: func(name string, err error) error {
				failures = append(failures, recordedFailure{name: name, err: err})
				return nil
			},
		})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}

		got, err := src.ListFrontmatters(t.Context())
		if err != nil {
			t.Fatalf("ListFrontmatters() failed: %v", err)
		}
		want := []*skill.Frontmatter{
			{Name: "alpha-valid", Description: "Valid alpha"},
			{Name: "zeta-valid", Description: "Valid zeta"},
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ListFrontmatters() mismatch (-want +got):\n%s", diff)
		}

		// The directory without SKILL.md is not reported.
		wantFailures := []recordedFailure{
			{name: "..", err: skill.ErrInvalidSkillName},
			{name: "bad-yaml", err: skill.ErrInvalidFrontmatter},
			{name: "mismatch", err: skill.ErrInvalidSkillName},
		}
		if len(failures) != len(wantFailures) {
			t.Fatalf("len(failures) = %d, want %d: %+v", len(failures), len(wantFailures), failures)
		}
		for i, want := range wantFailures {
			if failures[i].name != want.name || !errors.Is(failures[i].err, want.err) {
				t.Errorf("failures[%d] = (%q, %v), want (%q, %v)", i, failures[i].name, failures[i].err, want.name, want.err)
			}
		}
	})

	t.Run("aborts the listing with the error that the callback returns", func(t *testing.T) {
		sentinel := errors.New("caller aborted listing")
		calls := 0
		src, err := NewSource(Config{
			Client:   mustNewFakeGCS(t, objects).client,
			Bucket:   testBucket,
			BasePath: "skills",
			OnListInvalidSkill: func(name string, err error) error {
				calls++
				return fmt.Errorf("%w: on %s: %w", sentinel, name, err)
			},
		})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}

		_, err = src.ListFrontmatters(t.Context())
		if !errors.Is(err, sentinel) || !errors.Is(err, skill.ErrInvalidSkillName) {
			t.Fatalf("ListFrontmatters() = %v, want both the callback's error and ErrInvalidSkillName", err)
		}
		if calls != 1 {
			t.Errorf("callback calls = %d, want 1 (the listing must stop at the first abort)", calls)
		}
	})

	// A request that fails says nothing about the skill, so the listing
	// fails and the callback is not asked whether to skip it.
	for _, tc := range []struct {
		name string
		fail func(fake *fakeGCS)
	}{
		{name: "forbidden SKILL.md", fail: func(fake *fakeGCS) {
			fake.failRead("skills/alpha-valid/SKILL.md", http.StatusForbidden)
		}},
		{name: "connection lost in the middle of SKILL.md", fail: func(fake *fakeGCS) {
			fake.cutReadAfter("skills/alpha-valid/SKILL.md", len("---\nname: alpha-valid\n"))
		}},
	} {
		t.Run(tc.name+" fails the listing without calling OnListInvalidSkill", func(t *testing.T) {
			fake := mustNewFakeGCS(t, map[string][]byte{
				"skills/alpha-valid/SKILL.md": []byte("---\nname: alpha-valid\ndescription: Valid alpha\n---\nBody"),
			})
			tc.fail(fake)
			called := false
			src, err := NewSource(Config{
				Client:   fake.client,
				Bucket:   testBucket,
				BasePath: "skills",
				OnListInvalidSkill: func(string, error) error {
					called = true
					return nil
				},
			})
			if err != nil {
				t.Fatalf("NewSource() failed: %v", err)
			}

			if _, err := src.ListFrontmatters(t.Context()); !failedWith(err, http.StatusForbidden) {
				t.Fatalf("ListFrontmatters() = %v, want the error of the failed request", err)
			}
			if called {
				t.Error("OnListInvalidSkill was called for a failed request; want the listing to fail")
			}
		})
	}

	t.Run("LoadFrontmatter never calls OnListInvalidSkill", func(t *testing.T) {
		called := false
		src, err := NewSource(Config{
			Client:   mustNewFakeGCS(t, objects).client,
			Bucket:   testBucket,
			BasePath: "skills",
			OnListInvalidSkill: func(string, error) error {
				called = true
				return nil
			},
		})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}

		if _, err := src.LoadFrontmatter(t.Context(), "bad-yaml"); !errors.Is(err, skill.ErrInvalidFrontmatter) {
			t.Fatalf("LoadFrontmatter(bad-yaml) = %v, want %v", err, skill.ErrInvalidFrontmatter)
		}
		if called {
			t.Error("OnListInvalidSkill was called during LoadFrontmatter")
		}
	})
}

// TestGCSSource_DirectoryNames lists directories whose names no skill can
// have. Such a directory is ignored like any other while it holds no SKILL.md,
// and is an invalid skill once it does.
func TestGCSSource_DirectoryNames(t *testing.T) {
	alpha := []byte("---\nname: alpha\ndescription: Alpha\n---\nBody")

	t.Run("without SKILL.md", func(t *testing.T) {
		src, err := NewSource(Config{
			Client: mustNewFakeGCS(t, map[string][]byte{
				"skills/alpha/SKILL.md":   alpha,
				"skills/__pycache__/x.py": []byte("not a skill"),
				"skills/C:x/notes.txt":    []byte("not a skill"),
				"skills//notes.txt":       []byte("not a skill"),
				"skills/./notes.txt":      []byte("not a skill"),
				"skills/../notes.txt":     []byte("not a skill"),
				`skills/a\b/notes.txt`:    []byte("not a skill"),
			}).client,
			Bucket:   testBucket,
			BasePath: "skills",
		})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}

		got, err := src.ListFrontmatters(t.Context())
		if err != nil {
			t.Fatalf("ListFrontmatters() failed: %v", err)
		}
		want := []*skill.Frontmatter{{Name: "alpha", Description: "Alpha"}}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ListFrontmatters() mismatch (-want +got):\n%s", diff)
		}
	})

	for _, object := range []string{
		"skills//SKILL.md",
		"skills/./SKILL.md",
		"skills/../SKILL.md",
		`skills/my\skill/SKILL.md`,
		"skills/C:evil/SKILL.md",
	} {
		t.Run(fmt.Sprintf("with SKILL.md at %q", object), func(t *testing.T) {
			src, err := NewSource(Config{
				Client: mustNewFakeGCS(t, map[string][]byte{
					"skills/alpha/SKILL.md": alpha,
					object:                  []byte("---\nname: evil\ndescription: Evil\n---\nBody"),
				}).client,
				Bucket:   testBucket,
				BasePath: "skills",
			})
			if err != nil {
				t.Fatalf("NewSource() failed: %v", err)
			}

			if _, err := src.ListFrontmatters(t.Context()); !errors.Is(err, skill.ErrInvalidSkillName) {
				t.Fatalf("ListFrontmatters() = %v, want %v", err, skill.ErrInvalidSkillName)
			}
		})
	}
}

func TestGCSSource_ListPages(t *testing.T) {
	objects := map[string][]byte{
		"skills/alpha/SKILL.md":            []byte("---\nname: alpha\ndescription: Alpha skill\n---\nAlpha body"),
		"skills/alpha/references/ref-1.md": []byte("ref 1"),
		"skills/alpha/references/ref-2.md": []byte("ref 2"),
		"skills/alpha/references/ref-3.md": []byte("ref 3"),
		"skills/beta/SKILL.md":             []byte("---\nname: beta\ndescription: Beta skill\n---\nBeta body"),
		"skills/gamma/SKILL.md":            []byte("---\nname: gamma\ndescription: Gamma skill\n---\nGamma body"),
		"skills/not-a-skill/notes.txt":     []byte("ignored"),
	}
	wantFrontmatters := []*skill.Frontmatter{
		{Name: "alpha", Description: "Alpha skill"},
		{Name: "beta", Description: "Beta skill"},
		{Name: "gamma", Description: "Gamma skill"},
	}

	t.Run("listings of several pages", func(t *testing.T) {
		fake := mustNewFakeGCS(t, objects)
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket, BasePath: "skills"})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}
		fake.setPageSize(2)

		frontmatters, err := src.ListFrontmatters(t.Context())
		if err != nil {
			t.Fatalf("ListFrontmatters() failed: %v", err)
		}
		if diff := cmp.Diff(wantFrontmatters, frontmatters); diff != "" {
			t.Errorf("ListFrontmatters() mismatch (-want +got):\n%s", diff)
		}
		// Four directories at two a page.
		if got := fake.listRequests.Load(); got != 2 {
			t.Errorf("list requests of ListFrontmatters() = %d, want 2", got)
		}

		fake.listRequests.Store(0)
		resources, err := src.ListResources(t.Context(), "alpha", "references")
		if err != nil {
			t.Fatalf("ListResources(alpha, references) failed: %v", err)
		}
		wantResources := []string{"references/ref-1.md", "references/ref-2.md", "references/ref-3.md"}
		if diff := cmp.Diff(wantResources, resources); diff != "" {
			t.Errorf("ListResources(alpha, references) mismatch (-want +got):\n%s", diff)
		}
		// Three objects at two a page.
		if got := fake.listRequests.Load(); got != 2 {
			t.Errorf("list requests of ListResources() = %d, want 2", got)
		}
	})

	t.Run("directory listed on two pages is read once", func(t *testing.T) {
		fake := mustNewFakeGCS(t, objects)
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket, BasePath: "skills"})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}
		fake.setDirectoryPages([][]string{
			{"skills/alpha/", "skills/beta/"},
			{"skills/beta/", "skills/gamma/"},
		})

		frontmatters, err := src.ListFrontmatters(t.Context())
		if err != nil {
			t.Fatalf("ListFrontmatters() failed: %v", err)
		}
		if diff := cmp.Diff(wantFrontmatters, frontmatters); diff != "" {
			t.Errorf("ListFrontmatters() mismatch (-want +got):\n%s", diff)
		}
		if got := fake.readRequests.Load(); got != 3 {
			t.Errorf("SKILL.md reads = %d, want 3 (one for each directory)", got)
		}
	})

	t.Run("directories listed out of order are sorted", func(t *testing.T) {
		fake := mustNewFakeGCS(t, objects)
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket, BasePath: "skills"})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}
		fake.setDirectoryPages([][]string{{"skills/gamma/", "skills/alpha/", "skills/beta/"}})

		frontmatters, err := src.ListFrontmatters(t.Context())
		if err != nil {
			t.Fatalf("ListFrontmatters() failed: %v", err)
		}
		if diff := cmp.Diff(wantFrontmatters, frontmatters); diff != "" {
			t.Errorf("ListFrontmatters() mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestGCSSource_LoadFrontmatterAndInstructions(t *testing.T) {
	for _, tc := range []struct {
		name             string
		basePath         string
		skillName        string
		blobs            map[string][]byte
		wantFrontmatter  *skill.Frontmatter
		wantInstructions string
		wantErr          error
	}{
		{
			name:      "success with CRLF and dashes in description",
			basePath:  "skills",
			skillName: "my-skill",
			blobs: map[string][]byte{
				"skills/my-skill/SKILL.md": []byte("---\r\nname: my-skill\r\ndescription: Compare A --- B\r\nlicense: MIT\r\n---\r\nBody line 1\nBody line 2"),
			},
			wantFrontmatter: &skill.Frontmatter{
				Name:        "my-skill",
				Description: "Compare A --- B",
				License:     "MIT",
			},
			wantInstructions: "Body line 1\nBody line 2",
		},
		{
			name:      "white space around the instructions is removed",
			basePath:  "skills",
			skillName: "my-skill",
			blobs: map[string][]byte{
				"skills/my-skill/SKILL.md": []byte("---\nname: my-skill\ndescription: desc\n---\n\n  Body line 1\n\nBody line 2\n\n"),
			},
			wantFrontmatter:  &skill.Frontmatter{Name: "my-skill", Description: "desc"},
			wantInstructions: "Body line 1\n\nBody line 2",
		},
		{
			name:      "skill not found",
			basePath:  "skills",
			skillName: "missing",
			wantErr:   skill.ErrSkillNotFound,
		},
		{
			name:      "prefix without SKILL.md is not a skill",
			basePath:  "skills",
			skillName: "my-skill",
			blobs: map[string][]byte{
				"skills/my-skill/references/doc.md": []byte("orphan resource"),
			},
			wantErr: skill.ErrSkillNotFound,
		},
		{
			name:      "name mismatch",
			basePath:  "skills",
			skillName: "wrong-dir",
			blobs: map[string][]byte{
				"skills/wrong-dir/SKILL.md": []byte("---\nname: my-skill\ndescription: desc\n---\nBody"),
			},
			wantErr: skill.ErrInvalidSkillName,
		},
		{
			name:      "unclosed frontmatter",
			basePath:  "skills",
			skillName: "my-skill",
			blobs: map[string][]byte{
				"skills/my-skill/SKILL.md": []byte("---\nname: my-skill\ndescription: desc\n--- trailing text\nBody\n"),
			},
			wantErr: skill.ErrInvalidFrontmatter,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fake := mustNewFakeGCS(t, tc.blobs)
			src, err := NewSource(Config{Client: fake.client, Bucket: testBucket, BasePath: tc.basePath})
			if err != nil {
				t.Fatalf("NewSource() failed: %v", err)
			}

			fm, err := src.LoadFrontmatter(t.Context(), tc.skillName)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("LoadFrontmatter(%q) error = %v, want %v", tc.skillName, err, tc.wantErr)
			}
			instructions, instrErr := src.LoadInstructions(t.Context(), tc.skillName)
			if !errors.Is(instrErr, tc.wantErr) {
				t.Fatalf("LoadInstructions(%q) error = %v, want %v", tc.skillName, instrErr, tc.wantErr)
			}
			if tc.wantErr != nil {
				return
			}
			if diff := cmp.Diff(tc.wantFrontmatter, fm); diff != "" {
				t.Errorf("LoadFrontmatter(%q) mismatch (-want +got):\n%s", tc.skillName, diff)
			}
			if instructions != tc.wantInstructions {
				t.Errorf("LoadInstructions(%q) = %q, want %q", tc.skillName, instructions, tc.wantInstructions)
			}
		})
	}
}

func TestGCSSource_InvalidSkillNameIssuesNoRequest(t *testing.T) {
	// Place loadable skills both where path.Clean would land and at the literal
	// uncleaned object key so only an up-front rejection passes.
	blobs := map[string][]byte{
		"tenants/a/skills/public/SKILL.md":                 []byte("---\nname: public\ndescription: Public\n---\nPublic"),
		"tenants/b/skills/secrets/SKILL.md":                []byte("---\nname: secrets\ndescription: Secret\n---\nSecret"),
		"tenants/a/skills/../../b/skills/secrets/SKILL.md": []byte("---\nname: secrets\ndescription: Literal\n---\nLiteral"),
		"tenants/a/skills/SKILL.md":                        []byte("---\nname: skills\ndescription: Parent\n---\nParent"),
		"tenants/a/skills/team-a/my-skill/SKILL.md":        []byte("---\nname: my-skill\ndescription: Nested\n---\nNested"),
		"tenants/a/skills/team-a//my-skill/SKILL.md":       []byte("---\nname: my-skill\ndescription: DoubleSlash\n---\nBody"),
	}

	invalidNames := []string{
		"",
		".",
		"..",
		"../../other-tenant/secrets",
		"../../b/skills/secrets",
		"skills/../../other-tenant/secrets",
		"skills/..",
		"/public",
		"public/",
		"team-a/my-skill",
		"team-a//my-skill",
		`my\skill`,
		"my\x00skill",
	}

	for _, badName := range invalidNames {
		t.Run(fmt.Sprintf("%q", badName), func(t *testing.T) {
			fake := mustNewFakeGCS(t, blobs)
			src, err := NewSource(Config{Client: fake.client, Bucket: testBucket, BasePath: "tenants/a/skills"})
			if err != nil {
				t.Fatalf("NewSource() failed: %v", err)
			}

			if _, err := src.LoadFrontmatter(t.Context(), badName); !errors.Is(err, skill.ErrInvalidSkillName) {
				t.Errorf("LoadFrontmatter(%q) = %v, want %v", badName, err, skill.ErrInvalidSkillName)
			}
			if _, err := src.LoadInstructions(t.Context(), badName); !errors.Is(err, skill.ErrInvalidSkillName) {
				t.Errorf("LoadInstructions(%q) = %v, want %v", badName, err, skill.ErrInvalidSkillName)
			}
			if _, err := src.ListResources(t.Context(), badName, "."); !errors.Is(err, skill.ErrInvalidSkillName) {
				t.Errorf("ListResources(%q) = %v, want %v", badName, err, skill.ErrInvalidSkillName)
			}
			if _, err := src.LoadResource(t.Context(), badName, "references/doc.md"); !errors.Is(err, skill.ErrInvalidSkillName) {
				t.Errorf("LoadResource(%q) = %v, want %v", badName, err, skill.ErrInvalidSkillName)
			}

			if got := fake.listRequests.Load() + fake.readRequests.Load(); got != 0 {
				t.Errorf("requests for invalid name %q = %d, want 0", badName, got)
			}
		})
	}
}

func TestGCSSource_ResourcesAndBinaryBlobs(t *testing.T) {
	blobs := map[string][]byte{
		"skills/my-skill/SKILL.md":                       []byte("---\nname: my-skill\ndescription: Test\n---\nInstructions"),
		"skills/my-skill/references/ref1.md":             []byte("ref1 content"),
		"skills/my-skill/references/deep/deeper/note.md": []byte("nested note"),
		"skills/my-skill/references/deep-notes.md":       []byte("not inside deep"),
		"skills/my-skill/references/diagram.png":         pngHeader,
		"skills/my-skill/assets/logo.png":                pngHeader,
		"skills/my-skill/assets/templates/tmpl.txt":      []byte("template body"),
		"skills/my-skill/scripts/runtime/helper.py":      []byte("helper source"),
		"skills/my-skill/scripts/binary-helper":          pngHeader,
		"skills/my-skill/unauthorized/secret.txt":        []byte("ignored"),
		"skills/my-skill/references/empty-sub/":          {},
		"skills/my-skill/references/":                    {},
		"skills/other-skill/SKILL.md":                    []byte("---\nname: other-skill\ndescription: Other\n---\nOther"),
		"skills/other-skill/references/other-only.md":    []byte("other"),
	}

	fake := mustNewFakeGCS(t, blobs)
	src, err := NewSource(Config{Client: fake.client, Bucket: testBucket, BasePath: "skills"})
	if err != nil {
		t.Fatalf("NewSource() failed: %v", err)
	}

	for _, tc := range []struct {
		name    string
		subpath string
		want    []string
		wantErr error
	}{
		{
			name:    "root dot",
			subpath: ".",
			want: []string{
				"references/deep-notes.md",
				"references/deep/deeper/note.md",
				"references/diagram.png",
				"references/ref1.md",
				"assets/logo.png",
				"assets/templates/tmpl.txt",
				"scripts/binary-helper",
				"scripts/runtime/helper.py",
			},
		},
		{
			name:    "root empty",
			subpath: "",
			want: []string{
				"references/deep-notes.md",
				"references/deep/deeper/note.md",
				"references/diagram.png",
				"references/ref1.md",
				"assets/logo.png",
				"assets/templates/tmpl.txt",
				"scripts/binary-helper",
				"scripts/runtime/helper.py",
			},
		},
		{
			name:    "specific directory",
			subpath: "assets",
			want:    []string{"assets/logo.png", "assets/templates/tmpl.txt"},
		},
		{
			// "references/deep-notes.md" shares the prefix but is a sibling.
			name:    "nested subdirectory",
			subpath: "references/deep",
			want:    []string{"references/deep/deeper/note.md"},
		},
		{
			name:    "exact file subpath",
			subpath: "assets/logo.png",
			want:    []string{"assets/logo.png"},
		},
		{
			name:    "empty directory marker exists",
			subpath: "references/empty-sub",
			want:    nil,
		},
		{
			name:    "missing directory",
			subpath: "references/missing",
			wantErr: skill.ErrResourceNotFound,
		},
		{
			name:    "unauthorized directory",
			subpath: "unauthorized",
			wantErr: skill.ErrInvalidResourcePath,
		},
	} {
		t.Run("ListResources/"+tc.name, func(t *testing.T) {
			got, err := src.ListResources(t.Context(), "my-skill", tc.subpath)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("ListResources(%q) error = %v, want %v", tc.subpath, err, tc.wantErr)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ListResources(%q) mismatch (-want +got):\n%s", tc.subpath, diff)
			}
		})
	}

	for _, tc := range []struct {
		name         string
		skillName    string
		resourcePath string
		want         []byte
		wantErr      error
	}{
		{
			name:         "text reference",
			skillName:    "my-skill",
			resourcePath: "references/deep/deeper/note.md",
			want:         []byte("nested note"),
		},
		{
			name:         "binary asset",
			skillName:    "my-skill",
			resourcePath: "assets/logo.png",
			want:         pngHeader,
		},
		{
			name:         "binary script preserved as bytes",
			skillName:    "my-skill",
			resourcePath: "scripts/binary-helper",
			want:         pngHeader,
		},
		{
			name:         "cleanable relative path within allowed dir",
			skillName:    "my-skill",
			resourcePath: "assets/../assets/templates/./tmpl.txt",
			want:         []byte("template body"),
		},
		{
			name:         "directory marker is not loadable as a file",
			skillName:    "my-skill",
			resourcePath: "references/empty-sub",
			wantErr:      skill.ErrResourceNotFound,
		},
		{
			name:         "missing resource",
			skillName:    "my-skill",
			resourcePath: "assets/missing.png",
			wantErr:      skill.ErrResourceNotFound,
		},
		{
			name:         "unauthorized directory",
			skillName:    "my-skill",
			resourcePath: "unauthorized/secret.txt",
			wantErr:      skill.ErrInvalidResourcePath,
		},
		{
			name:         "traversal out of skill",
			skillName:    "my-skill",
			resourcePath: "../../etc/passwd",
			wantErr:      skill.ErrInvalidResourcePath,
		},
		{
			name:         "missing skill",
			skillName:    "missing-skill",
			resourcePath: "assets/logo.png",
			wantErr:      skill.ErrSkillNotFound,
		},
	} {
		t.Run("LoadResource/"+tc.name, func(t *testing.T) {
			rc, err := src.LoadResource(t.Context(), tc.skillName, tc.resourcePath)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("LoadResource(%q, %q) error = %v, want %v", tc.skillName, tc.resourcePath, err, tc.wantErr)
			}
			if err != nil {
				return
			}
			data, err := io.ReadAll(rc)
			if err := rc.Close(); err != nil {
				t.Fatalf("Close() failed: %v", err)
			}
			if err != nil {
				t.Fatalf("ReadAll() failed: %v", err)
			}
			if !bytes.Equal(data, tc.want) {
				t.Errorf("LoadResource(%q, %q) bytes mismatch: got %x, want %x", tc.skillName, tc.resourcePath, data, tc.want)
			}
		})
	}
}

// TestGCSSource_ListResourcesBucketOnlyShapes covers object layouts that a
// file system cannot hold.
func TestGCSSource_ListResourcesBucketOnlyShapes(t *testing.T) {
	fake := mustNewFakeGCS(t, map[string][]byte{
		"my-skill/SKILL.md": []byte("---\nname: my-skill\ndescription: Test\n---\nInstructions"),
		// A directory that exists only as its marker object.
		"my-skill/assets/": {},
		// Directories that exist only as ancestors of a marker object.
		"my-skill/scripts/tools/unused/": {},
		// A file and a directory of the same name.
		"my-skill/references/topic/details.md": []byte("details"),
		"my-skill/references/topic":            []byte("topic"),
		// A marker object that names a directory elsewhere declares nothing.
		"other-skill/SKILL.md":             []byte("---\nname: other-skill\ndescription: Test\n---\nInstructions"),
		"other-skill/assets/../elsewhere/": {},
	})
	src, err := NewSource(Config{Client: fake.client, Bucket: testBucket})
	if err != nil {
		t.Fatalf("NewSource() failed: %v", err)
	}

	for _, tc := range []struct {
		subpath string
		want    []string
	}{
		{subpath: "assets", want: nil},
		{subpath: "scripts", want: nil},
		{subpath: "scripts/tools", want: nil},
		{subpath: "references", want: []string{"references/topic", "references/topic/details.md"}},
	} {
		got, err := src.ListResources(t.Context(), "my-skill", tc.subpath)
		if err != nil {
			t.Fatalf("ListResources(%q) failed: %v", tc.subpath, err)
		}
		if diff := cmp.Diff(tc.want, got); diff != "" {
			t.Errorf("ListResources(%q) mismatch (-want +got):\n%s", tc.subpath, diff)
		}
	}

	if _, err := src.ListResources(t.Context(), "other-skill", "assets"); !errors.Is(err, skill.ErrResourceNotFound) {
		t.Errorf("ListResources(other-skill, assets) = %v, want %v", err, skill.ErrResourceNotFound)
	}
}

// TestGCSSource_ResourceObjectNames checks how the name of an object becomes
// the path of a resource. The rules are those of adk-python.
func TestGCSSource_ResourceObjectNames(t *testing.T) {
	skillMD := []byte("---\nname: my-skill\ndescription: Test\n---\nBody")

	// An object whose name is not in the form of a resource path is that
	// resource all the same.
	for _, tc := range []struct {
		object   string
		resource string
	}{
		{object: "skills/my-skill/references/./doc.md", resource: "references/doc.md"},
		{object: "skills/my-skill/references/sub/./doc.md", resource: "references/sub/doc.md"},
		{object: "skills/my-skill/references/sub//doc.md", resource: "references/sub/doc.md"},
		{object: `skills/my-skill/references/sub\doc.md`, resource: "references/sub/doc.md"},
		{object: `skills/my-skill/references/.\sub\doc.md`, resource: "references/sub/doc.md"},
		// A colon makes a drive only after the first character of the name.
		{object: "skills/my-skill/references/sub/c:doc.md", resource: "references/sub/c:doc.md"},
		{object: "skills/my-skill/references/ab:doc.md", resource: "references/ab:doc.md"},
	} {
		object, resource := tc.object, tc.resource
		t.Run(fmt.Sprintf("object %q is the resource %q", object, resource), func(t *testing.T) {
			fake := mustNewFakeGCS(t, map[string][]byte{
				"skills/my-skill/SKILL.md": skillMD,
				object:                     []byte("content"),
			})
			src, err := NewSource(Config{Client: fake.client, Bucket: testBucket, BasePath: "skills"})
			if err != nil {
				t.Fatalf("NewSource() failed: %v", err)
			}

			got, err := src.ListResources(t.Context(), "my-skill", "references")
			if err != nil {
				t.Fatalf("ListResources() failed: %v", err)
			}
			if diff := cmp.Diff([]string{resource}, got); diff != "" {
				t.Errorf("ListResources() mismatch (-want +got):\n%s", diff)
			}
			reader, err := src.LoadResource(t.Context(), "my-skill", resource)
			if err != nil {
				t.Fatalf("LoadResource(%q) failed: %v", resource, err)
			}
			content, err := io.ReadAll(reader)
			if closeErr := reader.Close(); closeErr != nil {
				t.Errorf("Close() failed: %v", closeErr)
			}
			if err != nil || string(content) != "content" {
				t.Errorf("LoadResource(%q) read (%q, %v), want (%q, nil)", resource, content, err, "content")
			}
			// The complete preload lists and loads every resource.
			if _, _, err := skill.WithCompletePreloadSource(t.Context(), src); err != nil {
				t.Errorf("WithCompletePreloadSource() failed: %v", err)
			}
		})
	}

	// An object whose name adk-python refuses makes the directory invalid.
	for _, rel := range []string{
		"../doc.md",
		"../../../../tmp/malicious.py",
		"a/..",
		"a/../doc.md",
		`a\..\..\evil.py`,
		`..\evil.py`,
		"/etc/passwd",
		`\etc\passwd`,
		".",
		"./.",
		"c:notes.md",
		`C:\evil.py`,
		"./c:notes.md",
		"1:notes.md",
		"é:notes.md",
	} {
		t.Run(fmt.Sprintf("object name %q is rejected", rel), func(t *testing.T) {
			fake := mustNewFakeGCS(t, map[string][]byte{
				"skills/my-skill/SKILL.md":            skillMD,
				"skills/my-skill/references/" + rel:   []byte("payload"),
				"skills/my-skill/references/guide.md": []byte("guide"),
			})
			src, err := NewSource(Config{Client: fake.client, Bucket: testBucket, BasePath: "skills"})
			if err != nil {
				t.Fatalf("NewSource() failed: %v", err)
			}

			if _, err := src.ListResources(t.Context(), "my-skill", "."); !errors.Is(err, skill.ErrInvalidResourcePath) {
				t.Errorf("ListResources(.) = %v, want %v", err, skill.ErrInvalidResourcePath)
			}
			if _, err := src.ListResources(t.Context(), "my-skill", "references"); !errors.Is(err, skill.ErrInvalidResourcePath) {
				t.Errorf("ListResources(references) = %v, want %v", err, skill.ErrInvalidResourcePath)
			}
			if _, err := src.LoadResource(t.Context(), "my-skill", "references/missing.md"); !errors.Is(err, skill.ErrInvalidResourcePath) {
				t.Errorf("LoadResource(references/missing.md) = %v, want %v", err, skill.ErrInvalidResourcePath)
			}
			if _, _, err := skill.WithCompletePreloadSource(t.Context(), src); !errors.Is(err, skill.ErrInvalidResourcePath) {
				t.Errorf("WithCompletePreloadSource() = %v, want %v", err, skill.ErrInvalidResourcePath)
			}
		})
	}

	t.Run("object named with a drive is not served under its own name", func(t *testing.T) {
		fake := mustNewFakeGCS(t, map[string][]byte{
			"skills/my-skill/SKILL.md":              skillMD,
			"skills/my-skill/references/c:notes.md": []byte("content"),
		})
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket, BasePath: "skills"})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}

		if _, err := src.LoadResource(t.Context(), "my-skill", "references/c:notes.md"); !errors.Is(err, skill.ErrResourceNotFound) {
			t.Errorf("LoadResource(references/c:notes.md) = %v, want %v", err, skill.ErrResourceNotFound)
		}
		if got := fake.readRequests.Load(); got != 1 {
			t.Errorf("read requests = %d, want 1 (the SKILL.md; the object itself must not be asked for)", got)
		}
	})

	t.Run("object deleted after it was listed is not found", func(t *testing.T) {
		fake := mustNewFakeGCS(t, map[string][]byte{
			"skills/my-skill/SKILL.md":            skillMD,
			"skills/my-skill/references/./doc.md": []byte("content"),
		})
		// The listing still has the object; a read answers that there is none.
		fake.failRead("skills/my-skill/references/./doc.md", http.StatusNotFound)
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket, BasePath: "skills"})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}

		if _, err := src.LoadResource(t.Context(), "my-skill", "references/doc.md"); !errors.Is(err, skill.ErrResourceNotFound) {
			t.Errorf("LoadResource(references/doc.md) = %v, want %v", err, skill.ErrResourceNotFound)
		}
	})

	t.Run("two objects that name the same resource are rejected", func(t *testing.T) {
		fake := mustNewFakeGCS(t, map[string][]byte{
			"skills/my-skill/SKILL.md":            skillMD,
			"skills/my-skill/references/doc.md":   []byte("first"),
			"skills/my-skill/references/./doc.md": []byte("second"),
		})
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket, BasePath: "skills"})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}

		if _, err := src.ListResources(t.Context(), "my-skill", "references"); !errors.Is(err, skill.ErrInvalidResourcePath) {
			t.Errorf("ListResources(references) = %v, want %v", err, skill.ErrInvalidResourcePath)
		}
		if _, _, err := skill.WithCompletePreloadSource(t.Context(), src); !errors.Is(err, skill.ErrInvalidResourcePath) {
			t.Errorf("WithCompletePreloadSource() = %v, want %v", err, skill.ErrInvalidResourcePath)
		}
		// LoadResource does not list the directory when an object has
		// exactly the name of the resource, so it serves that object.
		reader, err := src.LoadResource(t.Context(), "my-skill", "references/doc.md")
		if err != nil {
			t.Fatalf("LoadResource(references/doc.md) failed: %v", err)
		}
		content, err := io.ReadAll(reader)
		if closeErr := reader.Close(); closeErr != nil {
			t.Errorf("Close() failed: %v", closeErr)
		}
		if err != nil || string(content) != "first" {
			t.Errorf("LoadResource(references/doc.md) read (%q, %v), want (%q, nil)", content, err, "first")
		}
	})
}

// TestGCSSource_ResourcePathArguments checks the resource paths that a caller
// passes.
func TestGCSSource_ResourcePathArguments(t *testing.T) {
	// The object with a backslash is the resource "references/sub/doc.txt".
	// Its own name is not the path of a resource, and must not open it.
	fake := mustNewFakeGCS(t, map[string][]byte{
		"my-skill/SKILL.md":               []byte("---\nname: my-skill\ndescription: Test\n---\nBody"),
		"my-skill/references/doc.txt":     []byte("doc"),
		`my-skill/references/sub\doc.txt`: []byte("backslash"),
	})
	src, err := NewSource(Config{Client: fake.client, Bucket: testBucket})
	if err != nil {
		t.Fatalf("NewSource() failed: %v", err)
	}

	for _, tc := range []struct {
		path    string
		wantErr error
	}{
		{path: "", wantErr: skill.ErrInvalidResourcePath},
		{path: ".", wantErr: skill.ErrInvalidResourcePath},
		{path: "/", wantErr: skill.ErrInvalidResourcePath},
		{path: "references", wantErr: skill.ErrInvalidResourcePath},
		{path: "references/", wantErr: skill.ErrInvalidResourcePath},
		{path: "/references/doc.txt", wantErr: skill.ErrInvalidResourcePath},
		{path: `references\doc.txt`, wantErr: skill.ErrInvalidResourcePath},
		{path: "references/../../etc/passwd", wantErr: skill.ErrInvalidResourcePath},
		{path: "a/../../../passwords", wantErr: skill.ErrInvalidResourcePath},
		{path: "SKILL.md", wantErr: skill.ErrInvalidResourcePath},
		{path: `references/sub\doc.txt`, wantErr: skill.ErrResourceNotFound},
		// A ".." segment that stays inside the directory is resolved.
		{path: "references/sub/../doc.txt"},
		{path: "assets/../references/doc.txt"},
	} {
		reader, err := src.LoadResource(t.Context(), "my-skill", tc.path)
		if !errors.Is(err, tc.wantErr) {
			t.Errorf("LoadResource(%q) = %v, want %v", tc.path, err, tc.wantErr)
		}
		if err == nil {
			if err := reader.Close(); err != nil {
				t.Errorf("Close() of %q failed: %v", tc.path, err)
			}
		}
	}

	for _, tc := range []struct {
		subpath string
		wantErr error
	}{
		{subpath: "/", wantErr: skill.ErrInvalidResourcePath},
		{subpath: "/references", wantErr: skill.ErrInvalidResourcePath},
		{subpath: `references\sub`, wantErr: skill.ErrInvalidResourcePath},
		{subpath: "references/../../etc", wantErr: skill.ErrInvalidResourcePath},
		{subpath: "SKILL.md", wantErr: skill.ErrInvalidResourcePath},
		{subpath: `references/missing\sub`, wantErr: skill.ErrResourceNotFound},
		{subpath: "references/sub/.."},
	} {
		if _, err := src.ListResources(t.Context(), "my-skill", tc.subpath); !errors.Is(err, tc.wantErr) {
			t.Errorf("ListResources(%q) = %v, want %v", tc.subpath, err, tc.wantErr)
		}
	}
}

func TestGCSSource_SkillMDSizeLimit(t *testing.T) {
	// The limit that the package documents, written out and not taken from
	// the package's constant, so that the test fails if the constant changes.
	const limit = 10 << 20
	header := "---\nname: my-skill\ndescription: desc\n---\n"
	largest := append([]byte(header), bytes.Repeat([]byte("a"), limit-len(header))...)

	t.Run("instructions of a file of the maximum size are loaded", func(t *testing.T) {
		fake := mustNewFakeGCS(t, map[string][]byte{"my-skill/SKILL.md": largest})
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}

		got, err := src.LoadInstructions(t.Context(), "my-skill")
		if err != nil {
			t.Fatalf("LoadInstructions() failed: %v", err)
		}
		if want := limit - len(header); len(got) != want {
			t.Errorf("len(instructions) = %d, want %d", len(got), want)
		}
	})

	t.Run("instructions of a file one byte larger are not loaded", func(t *testing.T) {
		fake := mustNewFakeGCS(t, map[string][]byte{"my-skill/SKILL.md": append(slices.Clone(largest), 'b')})
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}

		if _, err := src.LoadInstructions(t.Context(), "my-skill"); err == nil {
			t.Error("LoadInstructions() succeeded, want an error")
		}
		// The size is checked only where the instructions are read.
		want := &skill.Frontmatter{Name: "my-skill", Description: "desc"}
		got, err := src.LoadFrontmatter(t.Context(), "my-skill")
		if err != nil {
			t.Fatalf("LoadFrontmatter() failed: %v", err)
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("LoadFrontmatter() mismatch (-want +got):\n%s", diff)
		}
		listed, err := src.ListFrontmatters(t.Context())
		if err != nil {
			t.Fatalf("ListFrontmatters() failed: %v", err)
		}
		if diff := cmp.Diff([]*skill.Frontmatter{want}, listed); diff != "" {
			t.Errorf("ListFrontmatters() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("frontmatter that does not end within the limit is invalid", func(t *testing.T) {
		unending := append([]byte("---\nname: my-skill\ndescription: "), bytes.Repeat([]byte("x"), limit)...)
		unending = append(unending, "\n---\nBody"...)
		fake := mustNewFakeGCS(t, map[string][]byte{"my-skill/SKILL.md": unending})
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}

		if _, err := src.LoadFrontmatter(t.Context(), "my-skill"); !errors.Is(err, skill.ErrInvalidFrontmatter) {
			t.Errorf("LoadFrontmatter() = %v, want %v", err, skill.ErrInvalidFrontmatter)
		}
	})
}

// TestGCSSource_ResourcesOnlyOfValidSkills checks that resources are served
// only for a valid skill, and that the source reads a SKILL.md once for that,
// not before every resource.
func TestGCSSource_ResourcesOnlyOfValidSkills(t *testing.T) {
	objects := map[string][]byte{
		"alpha/SKILL.md":                []byte("---\nname: alpha\ndescription: Alpha\n---\nAlpha body"),
		"alpha/references/one.md":       []byte("one"),
		"alpha/references/two.md":       []byte("two"),
		"beta/SKILL.md":                 []byte("---\nname: beta\ndescription: Beta\n---\nBeta body"),
		"beta/assets/logo.txt":          []byte("logo"),
		"not-a-skill/references/doc.md": []byte("of a directory without SKILL.md"),
		"broken/SKILL.md":               []byte("not frontmatter"),
		"broken/references/doc.md":      []byte("of an invalid skill"),
	}
	// load reads a resource to its end and returns the error.
	load := func(t *testing.T, src skill.Source, name, resourcePath string) error {
		t.Helper()
		reader, err := src.LoadResource(t.Context(), name, resourcePath)
		if err != nil {
			return err
		}
		_, err = io.ReadAll(reader)
		if closeErr := reader.Close(); closeErr != nil {
			t.Errorf("Close() failed: %v", closeErr)
		}
		return err
	}
	mustNewSource := func(t *testing.T, fake *fakeGCS) skill.Source {
		t.Helper()
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}
		return src
	}

	t.Run("SKILL.md is read once for many resources", func(t *testing.T) {
		fake := mustNewFakeGCS(t, objects)
		src := mustNewSource(t, fake)

		for _, resourcePath := range []string{"references/one.md", "references/two.md", "references/one.md"} {
			if err := load(t, src, "alpha", resourcePath); err != nil {
				t.Fatalf("LoadResource(alpha, %q) failed: %v", resourcePath, err)
			}
		}
		if _, err := src.ListResources(t.Context(), "alpha", "."); err != nil {
			t.Fatalf("ListResources(alpha, .) failed: %v", err)
		}
		// SKILL.md once, and the three resources.
		if got := fake.readRequests.Load(); got != 4 {
			t.Errorf("read requests = %d, want 4", got)
		}
	})

	t.Run("listing makes every listed skill known", func(t *testing.T) {
		fake := mustNewFakeGCS(t, objects)
		src, err := NewSource(Config{
			Client:             fake.client,
			Bucket:             testBucket,
			OnListInvalidSkill: func(string, error) error { return nil },
		})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}
		if _, err := src.ListFrontmatters(t.Context()); err != nil {
			t.Fatalf("ListFrontmatters() failed: %v", err)
		}

		fake.readRequests.Store(0)
		if err := load(t, src, "alpha", "references/one.md"); err != nil {
			t.Fatalf("LoadResource(alpha) failed: %v", err)
		}
		if err := load(t, src, "beta", "assets/logo.txt"); err != nil {
			t.Fatalf("LoadResource(beta) failed: %v", err)
		}
		// The two resources; no SKILL.md.
		if got := fake.readRequests.Load(); got != 2 {
			t.Errorf("read requests after the listing = %d, want 2", got)
		}
	})

	t.Run("complete preload reads each resource with one request", func(t *testing.T) {
		fake := mustNewFakeGCS(t, objects)
		fake.removeObjects("broken/") // So that the listing succeeds.
		src := mustNewSource(t, fake)

		if _, _, err := skill.WithCompletePreloadSource(t.Context(), src); err != nil {
			t.Fatalf("WithCompletePreloadSource() failed: %v", err)
		}
		// One SKILL.md for each of the three directories in the listing, the
		// instructions of the two skills, and the three resources.
		if got := fake.readRequests.Load(); got != 8 {
			t.Errorf("read requests = %d, want 8", got)
		}
		// The directories, and the three resource directories of each skill.
		if got := fake.listRequests.Load(); got != 7 {
			t.Errorf("list requests = %d, want 7", got)
		}
	})

	t.Run("directory that is not a valid skill serves nothing", func(t *testing.T) {
		fake := mustNewFakeGCS(t, objects)
		src := mustNewSource(t, fake)
		// Make other skills known first; it must not open the rest.
		if err := load(t, src, "alpha", "references/one.md"); err != nil {
			t.Fatalf("LoadResource(alpha) failed: %v", err)
		}

		for _, tc := range []struct {
			name    string
			wantErr error
		}{
			{name: "not-a-skill", wantErr: skill.ErrSkillNotFound},
			{name: "broken", wantErr: skill.ErrInvalidFrontmatter},
		} {
			// Twice: a refusal must not be remembered as a skill either.
			for range 2 {
				if err := load(t, src, tc.name, "references/doc.md"); !errors.Is(err, tc.wantErr) {
					t.Errorf("LoadResource(%s) = %v, want %v", tc.name, err, tc.wantErr)
				}
				if _, err := src.ListResources(t.Context(), tc.name, "."); !errors.Is(err, tc.wantErr) {
					t.Errorf("ListResources(%s) = %v, want %v", tc.name, err, tc.wantErr)
				}
			}
		}
	})

	t.Run("skill whose SKILL.md is gone is refused once that is seen", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			// readAgain makes the source read the SKILL.md of alpha again.
			readAgain func(t *testing.T, src skill.Source)
		}{
			{name: "by LoadFrontmatter", readAgain: func(t *testing.T, src skill.Source) {
				if _, err := src.LoadFrontmatter(t.Context(), "alpha"); !errors.Is(err, skill.ErrSkillNotFound) {
					t.Fatalf("LoadFrontmatter(alpha) = %v, want %v", err, skill.ErrSkillNotFound)
				}
			}},
			{name: "by LoadInstructions", readAgain: func(t *testing.T, src skill.Source) {
				if _, err := src.LoadInstructions(t.Context(), "alpha"); !errors.Is(err, skill.ErrSkillNotFound) {
					t.Fatalf("LoadInstructions(alpha) = %v, want %v", err, skill.ErrSkillNotFound)
				}
			}},
			{name: "by ListFrontmatters", readAgain: func(t *testing.T, src skill.Source) {
				// The listing passes over the directory, which still has
				// its resources and has no SKILL.md any more.
				if _, err := src.ListFrontmatters(t.Context()); err != nil {
					t.Fatalf("ListFrontmatters() failed: %v", err)
				}
			}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				fake := mustNewFakeGCS(t, objects)
				fake.removeObjects("broken/") // So that the listing succeeds.
				src := mustNewSource(t, fake)
				if err := load(t, src, "alpha", "references/one.md"); err != nil {
					t.Fatalf("LoadResource(alpha) failed: %v", err)
				}

				fake.removeObjects("alpha/SKILL.md")
				// Not seen yet: the source does not read SKILL.md again
				// before every resource.
				if err := load(t, src, "alpha", "references/one.md"); err != nil {
					t.Fatalf("LoadResource(alpha) before SKILL.md is read again failed: %v", err)
				}
				tc.readAgain(t, src)

				if err := load(t, src, "alpha", "references/one.md"); !errors.Is(err, skill.ErrSkillNotFound) {
					t.Errorf("LoadResource(alpha) = %v, want %v", err, skill.ErrSkillNotFound)
				}
			})
		}
	})

	t.Run("skill deleted with all its objects is not found after a listing", func(t *testing.T) {
		fake := mustNewFakeGCS(t, objects)
		src := mustNewSource(t, fake)
		if _, err := src.ListResources(t.Context(), "beta", "."); err != nil {
			t.Fatalf("ListResources(beta, .) failed: %v", err)
		}

		fake.removeObjects("beta/")
		fake.removeObjects("broken/") // So that the listing succeeds.
		if _, err := src.ListFrontmatters(t.Context()); err != nil {
			t.Fatalf("ListFrontmatters() failed: %v", err)
		}

		if _, err := src.ListResources(t.Context(), "beta", "."); !errors.Is(err, skill.ErrSkillNotFound) {
			t.Errorf("ListResources(beta, .) = %v, want %v", err, skill.ErrSkillNotFound)
		}
	})

	t.Run("failed read of SKILL.md makes the skill unknown", func(t *testing.T) {
		fake := mustNewFakeGCS(t, objects)
		src := mustNewSource(t, fake)
		if err := load(t, src, "alpha", "references/one.md"); err != nil {
			t.Fatalf("LoadResource(alpha) failed: %v", err)
		}

		fake.failRead("alpha/SKILL.md", http.StatusForbidden)
		if _, err := src.LoadFrontmatter(t.Context(), "alpha"); !failedWith(err, http.StatusForbidden) {
			t.Fatalf("LoadFrontmatter(alpha) = %v, want the error of the failed request", err)
		}

		if err := load(t, src, "alpha", "references/one.md"); !failedWith(err, http.StatusForbidden) {
			t.Errorf("LoadResource(alpha) = %v, want the error of the failed request", err)
		}
	})
}

// TestGCSSource_FailedRequests checks that a request that fails is returned as
// the storage client reports it, and not as a missing or an invalid skill.
func TestGCSSource_FailedRequests(t *testing.T) {
	objects := map[string][]byte{
		"alpha/SKILL.md":           []byte("---\nname: alpha\ndescription: Alpha\n---\nAlpha body"),
		"alpha/references/doc.txt": []byte("doc"),
		"beta/SKILL.md":            []byte("---\nname: beta\ndescription: Beta\n---\nBeta body"),
	}
	// callAll calls every method of the source and reports each error to check.
	callAll := func(ctx context.Context, src skill.Source, check func(method string, err error)) {
		_, err := src.ListFrontmatters(ctx)
		check("ListFrontmatters", err)
		_, err = src.LoadFrontmatter(ctx, "alpha")
		check("LoadFrontmatter", err)
		_, err = src.LoadInstructions(ctx, "alpha")
		check("LoadInstructions", err)
		_, err = src.ListResources(ctx, "alpha", ".")
		check("ListResources", err)
		_, err = src.LoadResource(ctx, "alpha", "references/doc.txt")
		check("LoadResource", err)
	}

	t.Run("canceled context", func(t *testing.T) {
		fake := mustNewFakeGCS(t, objects)
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}
		ctx, cancel := context.WithCancel(t.Context())
		cancel()

		callAll(ctx, src, func(method string, err error) {
			if !errors.Is(err, context.Canceled) {
				t.Errorf("%s(canceled) = %v, want %v", method, err, context.Canceled)
			}
		})
	})

	t.Run("forbidden SKILL.md", func(t *testing.T) {
		fake := mustNewFakeGCS(t, objects)
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}
		fake.failRead("alpha/SKILL.md", http.StatusForbidden)

		callAll(t.Context(), src, func(method string, err error) {
			if !failedWith(err, http.StatusForbidden) {
				t.Errorf("%s() = %v, want the error of the failed request", method, err)
			}
		})
	})

	t.Run("connection lost in the middle of SKILL.md", func(t *testing.T) {
		fake := mustNewFakeGCS(t, objects)
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}
		fake.cutReadAfter("alpha/SKILL.md", len("---\nname: alpha\n"))

		// The part that arrived does not parse; the error must still be the
		// failed request and not invalid frontmatter.
		_, err = src.LoadFrontmatter(t.Context(), "alpha")
		if !failedWith(err, http.StatusForbidden) || errors.Is(err, skill.ErrInvalidFrontmatter) {
			t.Errorf("LoadFrontmatter() = %v, want the error of the failed request", err)
		}
	})

	t.Run("connection lost in the middle of the instructions", func(t *testing.T) {
		fake := mustNewFakeGCS(t, objects)
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}
		fake.cutReadAfter("alpha/SKILL.md", len(objects["alpha/SKILL.md"])-1)

		if _, err := src.LoadInstructions(t.Context(), "alpha"); !failedWith(err, http.StatusForbidden) {
			t.Errorf("LoadInstructions() = %v, want the error of the failed request", err)
		}
	})

	t.Run("forbidden resource", func(t *testing.T) {
		fake := mustNewFakeGCS(t, objects)
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}
		fake.failRead("alpha/references/doc.txt", http.StatusForbidden)

		if _, err := src.LoadResource(t.Context(), "alpha", "references/doc.txt"); !failedWith(err, http.StatusForbidden) {
			t.Errorf("LoadResource() = %v, want the error of the failed request", err)
		}
	})

	t.Run("forbidden list page", func(t *testing.T) {
		fake := mustNewFakeGCS(t, objects)
		src, err := NewSource(Config{Client: fake.client, Bucket: testBucket})
		if err != nil {
			t.Fatalf("NewSource() failed: %v", err)
		}
		fake.setPageSize(1)

		// The two skills are on two pages; the second one fails.
		fake.failListPage(2)
		if _, err := src.ListFrontmatters(t.Context()); !failedWith(err, http.StatusForbidden) {
			t.Errorf("ListFrontmatters() = %v, want the error of the failed request", err)
		}
		fake.failListPage(1)
		if _, err := src.ListResources(t.Context(), "alpha", "."); !failedWith(err, http.StatusForbidden) {
			t.Errorf("ListResources() = %v, want the error of the failed request", err)
		}
	})
}

func TestGCSSource_ConcurrentUse(t *testing.T) {
	blobs := map[string][]byte{
		"skills/alpha/SKILL.md":           []byte("---\nname: alpha\ndescription: Alpha\n---\nAlpha instructions"),
		"skills/alpha/references/doc.txt": []byte("alpha doc"),
		"skills/alpha/assets/logo.png":    pngHeader,
		"skills/beta/SKILL.md":            []byte("---\nname: beta\ndescription: Beta\n---\nBeta instructions"),
		"skills/beta/scripts/run.sh":      []byte("echo beta"),
	}
	fake := mustNewFakeGCS(t, blobs)
	src, err := NewSource(Config{Client: fake.client, Bucket: testBucket, BasePath: "skills"})
	if err != nil {
		t.Fatalf("NewSource() failed: %v", err)
	}

	const workers = 24
	var wg sync.WaitGroup
	errs := make(chan error, workers*6)

	for i := range workers {
		wg.Go(func() {
			ctx := t.Context()

			if fms, err := src.ListFrontmatters(ctx); err != nil || len(fms) != 2 {
				errs <- fmt.Errorf("worker %d ListFrontmatters: len=%d err=%w", i, len(fms), err)
				return
			}
			if fm, err := src.LoadFrontmatter(ctx, "alpha"); err != nil || fm.Name != "alpha" {
				errs <- fmt.Errorf("worker %d LoadFrontmatter: %w", i, err)
				return
			}
			if body, err := src.LoadInstructions(ctx, "beta"); err != nil || body != "Beta instructions" {
				errs <- fmt.Errorf("worker %d LoadInstructions: body=%q err=%w", i, body, err)
				return
			}
			if res, err := src.ListResources(ctx, "alpha", "."); err != nil || len(res) != 2 {
				errs <- fmt.Errorf("worker %d ListResources: len=%d err=%w", i, len(res), err)
				return
			}
			rc, err := src.LoadResource(ctx, "alpha", "assets/logo.png")
			if err != nil {
				errs <- fmt.Errorf("worker %d LoadResource: %w", i, err)
				return
			}
			data, readErr := io.ReadAll(rc)
			_ = rc.Close()
			if readErr != nil || !bytes.Equal(data, pngHeader) {
				errs <- fmt.Errorf("worker %d ReadAll: %w", i, readErr)
				return
			}
			if _, _, err := skill.WithCompletePreloadSource(ctx, src); err != nil {
				errs <- fmt.Errorf("worker %d WithCompletePreloadSource: %w", i, err)
			}
		})
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		t.Error(err)
	}
}

// TestGCSSource_EqualsFileSystemSource checks that the source answers like the
// file-system source over the same tree of files.
//
// The two are known to differ in two answers, which this test leaves out:
//   - A skill name that is not the name of one directory, such as ".." or
//     "a/b", is [skill.ErrInvalidSkillName] here. The file-system source looks
//     for the path that the name makes and mostly returns
//     [skill.ErrSkillNotFound].
//   - LoadResource of a directory is [skill.ErrResourceNotFound] here. The
//     file-system source opens the directory without an error.
//
// The order of the resources is not compared. This source sorts the paths; the
// file-system source walks the directories, which gives another order where a
// name continues the name of a directory ("a/b.md" before "a-b.md" there,
// after it here).
func TestGCSSource_EqualsFileSystemSource(t *testing.T) {
	tree := map[string][]byte{
		"alpha/SKILL.md":                []byte("---\nname: alpha\ndescription: Alpha skill\nlicense: Apache-2.0\nallowed-tools: Bash(git:*) Read\nmetadata:\n  owner: team-a\n---\n\nAlpha instructions\nline 2\n"),
		"alpha/references/a.md":         []byte("a file"),
		"alpha/references/a/b.md":       []byte("a subdir b"),
		"alpha/references/a-b.md":       []byte("a hyphen b"),
		"alpha/references/guide.md":     []byte("guide text"),
		"alpha/references/sub/deep.md":  []byte("deep text"),
		"alpha/assets/logo.png":         pngHeader,
		"alpha/scripts/run.sh":          []byte("#!/bin/sh\necho ok\n"),
		"alpha/scripts/compiled.bin":    pngHeader,
		"alpha/unauthorized/hidden.txt": []byte("hidden"),
		"beta/SKILL.md":                 []byte("---\r\nname: beta\r\ndescription: Beta skill\r\n---\r\nBeta body"),
		"not-a-skill/notes.txt":         []byte("no SKILL.md"),
		"nested/ignored/SKILL.md":       []byte("---\nname: ignored\ndescription: Ignored\n---\nIgnored"),
		"README.md":                     []byte("root file"),
		"SKILL.md":                      []byte("---\nname: root\ndescription: Root\n---\nRoot"),
	}

	for _, basePath := range []string{"", "catalog/skills", "catalog/skills/"} {
		t.Run(fmt.Sprintf("basePath=%q", basePath), func(t *testing.T) {
			fsSrc, gcsSrc := mustBuildTwinSources(t, basePath, tree)
			ctx := t.Context()

			fsFMs, fsErr := fsSrc.ListFrontmatters(ctx)
			gcsFMs, gcsErr := gcsSrc.ListFrontmatters(ctx)
			if err := compareErrors(fsErr, gcsErr); err != nil {
				t.Errorf("ListFrontmatters(): %v", err)
			}
			if diff := cmp.Diff(fsFMs, gcsFMs, cmpopts.SortSlices(func(a, b *skill.Frontmatter) bool { return a.Name < b.Name })); diff != "" {
				t.Errorf("ListFrontmatters diff (-fs +gcs):\n%s", diff)
			}

			for _, skillName := range []string{"alpha", "beta", "missing", "not-a-skill"} {
				fsFM, fsErr := fsSrc.LoadFrontmatter(ctx, skillName)
				gcsFM, gcsErr := gcsSrc.LoadFrontmatter(ctx, skillName)
				if err := compareErrors(fsErr, gcsErr); err != nil {
					t.Errorf("LoadFrontmatter(%q): %v", skillName, err)
				}
				if diff := cmp.Diff(fsFM, gcsFM); diff != "" {
					t.Errorf("LoadFrontmatter(%q) diff (-fs +gcs):\n%s", skillName, diff)
				}

				fsInstr, fsErr := fsSrc.LoadInstructions(ctx, skillName)
				gcsInstr, gcsErr := gcsSrc.LoadInstructions(ctx, skillName)
				if err := compareErrors(fsErr, gcsErr); err != nil {
					t.Errorf("LoadInstructions(%q): %v", skillName, err)
				}
				if fsInstr != gcsInstr {
					t.Errorf("LoadInstructions(%q): fs=%q, gcs=%q", skillName, fsInstr, gcsInstr)
				}
			}

			for _, subpath := range []string{"", ".", "references", "references/", "references/sub", "references/sub/..", "references/../assets", "assets", "scripts", "assets/logo.png", "references/missing", `references/sub\x`, "unauthorized", "references/../../x", "/references"} {
				fsRes, fsErr := fsSrc.ListResources(ctx, "alpha", subpath)
				gcsRes, gcsErr := gcsSrc.ListResources(ctx, "alpha", subpath)
				if err := compareErrors(fsErr, gcsErr); err != nil {
					t.Errorf("ListResources(alpha, %q): %v", subpath, err)
				}
				if diff := cmp.Diff(fsRes, gcsRes, cmpopts.SortSlices(strings.Compare)); diff != "" {
					t.Errorf("ListResources(alpha, %q) diff (-fs +gcs):\n%s", subpath, diff)
				}
			}
			fsBetaRes, fsErr := fsSrc.ListResources(ctx, "beta", ".")
			gcsBetaRes, gcsErr := gcsSrc.ListResources(ctx, "beta", ".")
			if err := compareErrors(fsErr, gcsErr); err != nil {
				t.Errorf("ListResources(beta, .): %v", err)
			}
			if diff := cmp.Diff(fsBetaRes, gcsBetaRes); diff != "" {
				t.Errorf("ListResources(beta, .) diff (-fs +gcs):\n%s", diff)
			}

			for _, resPath := range []string{
				"references/a.md",
				"references/a/b.md",
				"references/a-b.md",
				"references/guide.md",
				"references/sub/deep.md",
				"assets/logo.png",
				"scripts/run.sh",
				"scripts/compiled.bin",
				"assets/../assets/./logo.png",
				"references/sub/../guide.md",
				"assets/missing.png",
				"references/a.md/x",
				`references/sub\deep.md`,
				"unauthorized/hidden.txt",
				"../../etc/passwd",
				"references/../../../passwords",
				"references",
				"/references/a.md",
				"",
			} {
				fsRC, fsErr := fsSrc.LoadResource(ctx, "alpha", resPath)
				gcsRC, gcsErr := gcsSrc.LoadResource(ctx, "alpha", resPath)
				if err := compareErrors(fsErr, gcsErr); err != nil {
					t.Errorf("LoadResource(alpha, %q): %v", resPath, err)
				}
				if fsErr != nil || gcsErr != nil {
					continue
				}
				fsBytes, _ := io.ReadAll(fsRC)
				_ = fsRC.Close()
				gcsBytes, _ := io.ReadAll(gcsRC)
				_ = gcsRC.Close()
				if !bytes.Equal(fsBytes, gcsBytes) {
					t.Errorf("LoadResource(alpha, %q): fs=%x, gcs=%x", resPath, fsBytes, gcsBytes)
				}
			}
		})
	}

	for _, tc := range []struct {
		name      string
		skillName string
		tree      map[string][]byte
	}{
		{
			name:      "invalid frontmatter tree",
			skillName: "broken",
			tree: map[string][]byte{
				"broken/SKILL.md": []byte("---\n[bad yaml"),
			},
		},
		{
			name:      "mismatched skill name tree",
			skillName: "wrong-dir",
			tree: map[string][]byte{
				"wrong-dir/SKILL.md": []byte("---\nname: right-name\ndescription: desc\n---\nBody"),
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fsSrc, gcsSrc := mustBuildTwinSources(t, "skills", tc.tree)
			ctx := t.Context()

			_, fsErr := fsSrc.ListFrontmatters(ctx)
			_, gcsErr := gcsSrc.ListFrontmatters(ctx)
			if err := compareErrors(fsErr, gcsErr); err != nil {
				t.Errorf("ListFrontmatters(): %v", err)
			}

			_, fsErr = fsSrc.LoadFrontmatter(ctx, tc.skillName)
			_, gcsErr = gcsSrc.LoadFrontmatter(ctx, tc.skillName)
			if err := compareErrors(fsErr, gcsErr); err != nil {
				t.Errorf("LoadFrontmatter(): %v", err)
			}

			_, fsErr = fsSrc.LoadInstructions(ctx, tc.skillName)
			_, gcsErr = gcsSrc.LoadInstructions(ctx, tc.skillName)
			if err := compareErrors(fsErr, gcsErr); err != nil {
				t.Errorf("LoadInstructions(): %v", err)
			}

			_, fsErr = fsSrc.ListResources(ctx, tc.skillName, ".")
			_, gcsErr = gcsSrc.ListResources(ctx, tc.skillName, ".")
			if err := compareErrors(fsErr, gcsErr); err != nil {
				t.Errorf("ListResources(): %v", err)
			}

			_, fsErr = fsSrc.LoadResource(ctx, tc.skillName, "references/doc.md")
			_, gcsErr = gcsSrc.LoadResource(ctx, tc.skillName, "references/doc.md")
			if err := compareErrors(fsErr, gcsErr); err != nil {
				t.Errorf("LoadResource(): %v", err)
			}
		})
	}
}

// mustBuildTwinSources returns a file-system source and a Cloud Storage source
// over the same tree of files, the latter with the tree under basePath, or
// fails the test.
func mustBuildTwinSources(t *testing.T, basePath string, tree map[string][]byte) (skill.Source, skill.Source) {
	t.Helper()
	mapFS := make(fstest.MapFS, len(tree))
	for k, v := range tree {
		mapFS[k] = &fstest.MapFile{Data: v}
	}

	prefix := ""
	if basePath != "" {
		prefix = strings.TrimSuffix(basePath, "/") + "/"
	}
	gcsBlobs := make(map[string][]byte, len(tree))
	if prefix == "" {
		maps.Copy(gcsBlobs, tree)
	} else {
		for k, v := range tree {
			gcsBlobs[prefix+k] = v
		}
	}
	fake := mustNewFakeGCS(t, gcsBlobs)
	gcsSrc, err := NewSource(Config{Client: fake.client, Bucket: testBucket, BasePath: basePath})
	if err != nil {
		t.Fatalf("NewSource() failed: %v", err)
	}
	return skill.NewFileSystemSource(mapFS), gcsSrc
}

// compareErrors returns an error that says how the errors of the two sources
// differ, or nil if they do not: both are nil, or both wrap the same errors of
// package skill.
func compareErrors(fsErr, gcsErr error) error {
	if (fsErr == nil) != (gcsErr == nil) {
		return fmt.Errorf("file-system source returned error %v, Cloud Storage source %v", fsErr, gcsErr)
	}
	if fsErr == nil {
		return nil
	}
	wrapsAny := false
	for _, sentinel := range []error{
		skill.ErrInvalidSkillName,
		skill.ErrInvalidFrontmatter,
		skill.ErrSkillNotFound,
		skill.ErrDuplicateSkill,
		skill.ErrInvalidResourcePath,
		skill.ErrResourceNotFound,
	} {
		if errors.Is(fsErr, sentinel) != errors.Is(gcsErr, sentinel) {
			return fmt.Errorf("only one of the errors is %q: file-system source returned %v, Cloud Storage source %v", sentinel, fsErr, gcsErr)
		}
		wrapsAny = wrapsAny || errors.Is(fsErr, sentinel)
	}
	if !wrapsAny {
		return fmt.Errorf("neither error is one of package skill: file-system source returned %v, Cloud Storage source %v", fsErr, gcsErr)
	}
	return nil
}
