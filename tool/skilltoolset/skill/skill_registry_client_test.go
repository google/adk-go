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

package skill

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/option"
)

// recordedRequest is a request captured by the fake Skill Registry.
type recordedRequest struct {
	path  string
	query url.Values
}

// newTestClient starts a fake Skill Registry that replies with the given
// handler and returns a client pointed at it, plus the requests it received.
func newTestClient(t *testing.T, handler http.HandlerFunc) (*SkillRegistryClient, *[]recordedRequest) {
	t.Helper()

	var got []recordedRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = append(got, recordedRequest{path: r.URL.Path, query: r.URL.Query()})
		handler(w, r)
	}))
	t.Cleanup(srv.Close)

	c, err := NewSkillRegistryClient(t.Context(),
		SkillRegistryClientConfig{ProjectID: "p", Location: "us-central1"},
		option.WithHTTPClient(srv.Client()),
		option.WithEndpoint(srv.URL),
	)
	if err != nil {
		t.Fatalf("NewSkillRegistryClient() error = %v", err)
	}
	return c, &got
}

// jsonHandler replies with body for every request.
func jsonHandler(body string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}
}

func TestNewSkillRegistryClient_Config(t *testing.T) {
	tests := []struct {
		name    string
		cfg     SkillRegistryClientConfig
		wantErr bool
	}{
		{name: "ok", cfg: SkillRegistryClientConfig{ProjectID: "p", Location: "us-central1"}},
		{name: "missing project", cfg: SkillRegistryClientConfig{Location: "us-central1"}, wantErr: true},
		{name: "missing location", cfg: SkillRegistryClientConfig{ProjectID: "p"}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// A caller-supplied HTTP client keeps the constructor off ADC.
			c, err := NewSkillRegistryClient(t.Context(), tt.cfg, option.WithHTTPClient(http.DefaultClient))
			if gotErr := err != nil; gotErr != tt.wantErr {
				t.Fatalf("NewSkillRegistryClient() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if want := "projects/p/locations/us-central1"; c.parent != want {
				t.Errorf("parent = %q, want %q", c.parent, want)
			}
			if want := "https://us-central1-aiplatform.googleapis.com/v1beta1"; c.baseURL != want {
				t.Errorf("baseURL = %q, want %q", c.baseURL, want)
			}
		})
	}
}

func TestSkillRegistryClient_ListSkills(t *testing.T) {
	body := `{
	  "skills": [
	    {
	      "name": "projects/1234/locations/us-central1/skills/cymbal-skill",
	      "displayName": "cymbal_skill",
	      "description": "A skill for managing Cymbal projects.",
	      "state": "ACTIVE",
	      "createTime": "2026-05-10T00:02:12.497720Z",
	      "updateTime": "2026-05-10T00:02:19.064874Z"
	    }
	  ],
	  "nextPageToken": "next"
	}`
	c, got := newTestClient(t, jsonHandler(body))

	resp, err := c.ListSkills(t.Context(), WithPageSize(25), WithPageToken("tok"))
	if err != nil {
		t.Fatalf("ListSkills() error = %v", err)
	}

	want := &ListSkillsResponse{
		Skills: []Skill{{
			Name:        "projects/1234/locations/us-central1/skills/cymbal-skill",
			DisplayName: "cymbal_skill",
			Description: "A skill for managing Cymbal projects.",
			State:       SkillStateActive,
			CreateTime:  time.Date(2026, 5, 10, 0, 2, 12, 497720000, time.UTC),
			UpdateTime:  time.Date(2026, 5, 10, 0, 2, 19, 64874000, time.UTC),
		}},
		NextPageToken: "next",
	}
	if diff := cmp.Diff(want, resp); diff != "" {
		t.Errorf("ListSkills() diff (-want +got):\n%s", diff)
	}
	if id := resp.Skills[0].ID(); id != "cymbal-skill" {
		t.Errorf("Skill.ID() = %q, want %q", id, "cymbal-skill")
	}

	req := (*got)[0]
	if want := "/projects/p/locations/us-central1/skills"; req.path != want {
		t.Errorf("path = %q, want %q", req.path, want)
	}
	if want := (url.Values{"pageSize": {"25"}, "pageToken": {"tok"}}); !cmp.Equal(want, req.query) {
		t.Errorf("query = %v, want %v", req.query, want)
	}
}

func TestSkillRegistryClient_AllSkills_Paging(t *testing.T) {
	pageOne := `{"skills":[{"displayName":"one"},{"displayName":"two"}],"nextPageToken":"p2"}`
	pageTwo := `{"skills":[{"displayName":"three"}]}`
	c, got := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("pageToken") == "p2" {
			_, _ = w.Write([]byte(pageTwo))
			return
		}
		_, _ = w.Write([]byte(pageOne))
	})

	var names []string
	for sk, err := range c.AllSkills(t.Context(), WithPageSize(2)) {
		if err != nil {
			t.Fatalf("AllSkills() error = %v", err)
		}
		names = append(names, sk.DisplayName)
	}

	if want := []string{"one", "two", "three"}; !slices.Equal(names, want) {
		t.Errorf("AllSkills() names = %v, want %v", names, want)
	}
	if len(*got) != 2 {
		t.Fatalf("got %d requests, want 2", len(*got))
	}
	// The page size option must survive onto the follow-up page request.
	if size := (*got)[1].query.Get("pageSize"); size != "2" {
		t.Errorf("second request pageSize = %q, want %q", size, "2")
	}
}

func TestSkillRegistryClient_AllSkills_Error(t *testing.T) {
	c, _ := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":{"status":"PERMISSION_DENIED","message":"nope"}}`, http.StatusForbidden)
	})

	var iterations int
	for sk, err := range c.AllSkills(t.Context()) {
		iterations++
		if err == nil {
			t.Fatalf("AllSkills() yielded skill %v, want error", sk)
		}
		var apiErr *APIError
		if !errors.As(err, &apiErr) {
			t.Fatalf("AllSkills() error = %v, want *APIError", err)
		}
		if apiErr.Status != "PERMISSION_DENIED" || apiErr.Message != "nope" {
			t.Errorf("APIError = %+v, want status PERMISSION_DENIED and message nope", apiErr)
		}
	}
	if iterations != 1 {
		t.Errorf("AllSkills() yielded %d times, want 1", iterations)
	}
}

func TestSkillRegistryClient_GetSkill(t *testing.T) {
	zipped := []byte("PK\x03\x04 not really a zip \xff\xfe")

	tests := []struct {
		name     string
		encoded  string
		nameOrID string
		wantPath string
	}{
		{
			name:     "skill ID and standard base64",
			encoded:  base64.StdEncoding.EncodeToString(zipped),
			nameOrID: "cymbal-skill",
			wantPath: "/projects/p/locations/us-central1/skills/cymbal-skill",
		},
		{
			name:     "full resource name and URL-safe base64",
			encoded:  base64.RawURLEncoding.EncodeToString(zipped),
			nameOrID: "projects/1234/locations/us-central1/skills/cymbal-skill",
			wantPath: "/projects/1234/locations/us-central1/skills/cymbal-skill",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{
				"name":             "projects/1234/locations/us-central1/skills/cymbal-skill",
				"displayName":      "cymbal_skill",
				"zippedFilesystem": tt.encoded,
				"license":          "Apache-2.0",
				"labels":           map[string]string{"team": "cymbal"},
				"skillSource":      SkillSourceUser,
				"sha256":           "abc123",
			})
			if err != nil {
				t.Fatalf("marshaling response: %v", err)
			}
			c, got := newTestClient(t, jsonHandler(string(body)))

			sk, err := c.GetSkill(t.Context(), tt.nameOrID)
			if err != nil {
				t.Fatalf("GetSkill() error = %v", err)
			}
			if diff := cmp.Diff(zipped, []byte(sk.ZippedFilesystem)); diff != "" {
				t.Errorf("ZippedFilesystem diff (-want +got):\n%s", diff)
			}
			if sk.License != "Apache-2.0" || sk.SkillSource != SkillSourceUser || sk.SHA256 != "abc123" {
				t.Errorf("GetSkill() = %+v, want license/source/sha256 populated", sk)
			}
			if diff := cmp.Diff(map[string]string{"team": "cymbal"}, sk.Labels); diff != "" {
				t.Errorf("Labels diff (-want +got):\n%s", diff)
			}
			if (*got)[0].path != tt.wantPath {
				t.Errorf("path = %q, want %q", (*got)[0].path, tt.wantPath)
			}
		})
	}
}

func TestSkillRegistryClient_RetrieveSkills(t *testing.T) {
	body := `{"retrievedSkills":[
	  {"skillName":"projects/1234/locations/us-central1/skills/cymbal-skill","description":"Manage Cymbal."}
	]}`
	c, got := newTestClient(t, jsonHandler(body))

	resp, err := c.RetrieveSkills(t.Context(), "manage cloud resources", 5)
	if err != nil {
		t.Fatalf("RetrieveSkills() error = %v", err)
	}

	want := &RetrieveSkillsResponse{RetrievedSkills: []RetrievedSkill{{
		SkillName:   "projects/1234/locations/us-central1/skills/cymbal-skill",
		Description: "Manage Cymbal.",
	}}}
	if diff := cmp.Diff(want, resp); diff != "" {
		t.Errorf("RetrieveSkills() diff (-want +got):\n%s", diff)
	}
	if id := resp.RetrievedSkills[0].ID(); id != "cymbal-skill" {
		t.Errorf("RetrievedSkill.ID() = %q, want %q", id, "cymbal-skill")
	}

	req := (*got)[0]
	if want := "/projects/p/locations/us-central1/skills:retrieve"; req.path != want {
		t.Errorf("path = %q, want %q", req.path, want)
	}
	if want := (url.Values{"query": {"manage cloud resources"}, "topK": {"5"}}); !cmp.Equal(want, req.query) {
		t.Errorf("query = %v, want %v", req.query, want)
	}
}

func TestSkillRegistryClient_RetrieveSkills_DefaultTopK(t *testing.T) {
	c, got := newTestClient(t, jsonHandler(`{}`))

	if _, err := c.RetrieveSkills(t.Context(), "anything", 0); err != nil {
		t.Fatalf("RetrieveSkills() error = %v", err)
	}
	if _, ok := (*got)[0].query["topK"]; ok {
		t.Errorf("query = %v, want no topK so the server default applies", (*got)[0].query)
	}
}

func TestSkillRegistryClient_Revisions(t *testing.T) {
	listBody := `{"skillRevisions":[{
	  "name":"projects/1234/locations/us-central1/skills/cymbal-skill/revisions/4567",
	  "state":"ACTIVE",
	  "createTime":"2026-05-10T00:02:12.497720Z"
	}]}`
	c, got := newTestClient(t, jsonHandler(listBody))

	list, err := c.ListSkillRevisions(t.Context(), "cymbal-skill", WithFilter(`labels.env="prod"`))
	if err != nil {
		t.Fatalf("ListSkillRevisions() error = %v", err)
	}
	want := &ListSkillRevisionsResponse{SkillRevisions: []SkillRevision{{
		Name:       "projects/1234/locations/us-central1/skills/cymbal-skill/revisions/4567",
		State:      SkillStateActive,
		CreateTime: time.Date(2026, 5, 10, 0, 2, 12, 497720000, time.UTC),
	}}}
	if diff := cmp.Diff(want, list); diff != "" {
		t.Errorf("ListSkillRevisions() diff (-want +got):\n%s", diff)
	}
	if id := list.SkillRevisions[0].ID(); id != "4567" {
		t.Errorf("SkillRevision.ID() = %q, want %q", id, "4567")
	}

	req := (*got)[0]
	if want := "/projects/p/locations/us-central1/skills/cymbal-skill/revisions"; req.path != want {
		t.Errorf("path = %q, want %q", req.path, want)
	}
	if want := `labels.env="prod"`; req.query.Get("filter") != want {
		t.Errorf("filter = %q, want %q", req.query.Get("filter"), want)
	}

	if _, err := c.GetSkillRevision(t.Context(), "cymbal-skill", "4567"); err != nil {
		t.Fatalf("GetSkillRevision() error = %v", err)
	}
	req = (*got)[1]
	if want := "/projects/p/locations/us-central1/skills/cymbal-skill/revisions/4567"; req.path != want {
		t.Errorf("path = %q, want %q", req.path, want)
	}
}

func TestSkillRegistryClient_NotFound(t *testing.T) {
	c, _ := newTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":{"status":"NOT_FOUND","message":"Skill not found."}}`))
	})

	_, err := c.GetSkill(t.Context(), "missing-skill")
	if !errors.Is(err, ErrSkillNotFound) {
		t.Errorf("GetSkill() error = %v, want it to wrap ErrSkillNotFound", err)
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("GetSkill() error = %v, want *APIError", err)
	}
	if apiErr.StatusCode != http.StatusNotFound || apiErr.Status != "NOT_FOUND" {
		t.Errorf("APIError = %+v, want 404 NOT_FOUND", apiErr)
	}
}

func TestSkillRegistryClient_InvalidNames(t *testing.T) {
	c, got := newTestClient(t, jsonHandler(`{}`))

	tests := []struct {
		name string
		call func() error
	}{
		{
			name: "empty skill ID",
			call: func() error { _, err := c.GetSkill(t.Context(), ""); return err },
		},
		{
			name: "relative path as skill ID",
			call: func() error { _, err := c.GetSkill(t.Context(), "a/b"); return err },
		},
		{
			name: "malformed resource name",
			call: func() error { _, err := c.GetSkill(t.Context(), "projects/p/skills/s"); return err },
		},
		{
			name: "revision name instead of skill name",
			call: func() error {
				_, err := c.ListSkillRevisions(t.Context(), "projects/p/locations/l/skills/s/revisions/1")
				return err
			},
		},
		{
			name: "empty revision ID",
			call: func() error { _, err := c.GetSkillRevision(t.Context(), "s", ""); return err },
		},
		{
			name: "path as revision ID",
			call: func() error { _, err := c.GetSkillRevision(t.Context(), "s", "1/2"); return err },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.call(); !errors.Is(err, ErrInvalidSkillName) {
				t.Errorf("error = %v, want it to wrap ErrInvalidSkillName", err)
			}
		})
	}
	if len(*got) != 0 {
		t.Errorf("got %d requests, want none: invalid names must not reach the server", len(*got))
	}
}

func TestBase64Bytes(t *testing.T) {
	raw := []byte{0xff, 0xfe, 0x00, 0x3f, 0x3e, 'a'}

	tests := []struct {
		name    string
		json    string
		want    []byte
		wantErr bool
	}{
		{name: "standard padded", json: `"` + base64.StdEncoding.EncodeToString(raw) + `"`, want: raw},
		{name: "standard unpadded", json: `"` + base64.RawStdEncoding.EncodeToString(raw) + `"`, want: raw},
		{name: "url-safe padded", json: `"` + base64.URLEncoding.EncodeToString(raw) + `"`, want: raw},
		{name: "url-safe unpadded", json: `"` + base64.RawURLEncoding.EncodeToString(raw) + `"`, want: raw},
		{name: "empty", json: `""`},
		{name: "null", json: `null`},
		{name: "not a string", json: `123`, wantErr: true},
		{name: "not base64", json: `"!!!!"`, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got Base64Bytes
			err := json.Unmarshal([]byte(tt.json), &got)
			if gotErr := err != nil; gotErr != tt.wantErr {
				t.Fatalf("Unmarshal() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}
			if diff := cmp.Diff(tt.want, []byte(got)); diff != "" {
				t.Errorf("Unmarshal() diff (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("round trip", func(t *testing.T) {
		encoded, err := json.Marshal(Base64Bytes(raw))
		if err != nil {
			t.Fatalf("Marshal() error = %v", err)
		}
		if want := `"` + base64.StdEncoding.EncodeToString(raw) + `"`; string(encoded) != want {
			t.Errorf("Marshal() = %s, want %s", encoded, want)
		}
		var got Base64Bytes
		if err := json.Unmarshal(encoded, &got); err != nil {
			t.Fatalf("Unmarshal() error = %v", err)
		}
		if diff := cmp.Diff(raw, []byte(got)); diff != "" {
			t.Errorf("round trip diff (-want +got):\n%s", diff)
		}
	})

	t.Run("nil marshals to null", func(t *testing.T) {
		encoded, err := json.Marshal(Base64Bytes(nil))
		if err != nil {
			t.Fatalf("Marshal() error = %v", err)
		}
		if string(encoded) != "null" {
			t.Errorf("Marshal(nil) = %s, want null", encoded)
		}
	})
}
