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

// This file holds a read-only REST client for the Skill Registry of the Gemini
// Enterprise Agent Platform (aiplatform.googleapis.com, v1beta1). It covers the
// discovery methods — ListSkills, GetSkill, RetrieveSkills, ListSkillRevisions
// and GetSkillRevision — and deliberately omits the mutating ones (CreateSkill,
// UpdateSkill, DeleteSkill), which are long-running operations.
//
// See https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/skill-registry/create-manage.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"

	"google.golang.org/api/option"
	"google.golang.org/api/option/internaloption"
	htransport "google.golang.org/api/transport"
)

// cloudPlatformScope is the OAuth scope requested for Application Default
// Credentials.
const cloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

// skillRegistryAPIVersion is the Agent Platform API version that serves the
// Skill Registry. It is a Preview API, hence v1beta1.
const skillRegistryAPIVersion = "v1beta1"

// Skill states, as reported by [Skill.State] and [SkillRevision.State].
const (
	SkillStateUnspecified = "STATE_UNSPECIFIED"
	SkillStateActive      = "ACTIVE"
	SkillStateCreating    = "CREATING"
	SkillStateFailed      = "FAILED"
	SkillStateDeleting    = "DELETING"
)

// Skill origins, as reported by [Skill.SkillSource].
const (
	SkillSourceUnspecified = "SKILL_SOURCE_UNSPECIFIED"
	SkillSourceUser        = "USER"
	SkillSourceSystem      = "SYSTEM"
)

// Skill is a skill stored in the Skill Registry.
type Skill struct {
	// Name is the resource name of the skill, in the form
	// "projects/{project}/locations/{location}/skills/{skill}".
	Name string `json:"name,omitempty"`
	// DisplayName is the human-readable name of the skill; it mirrors the
	// "name" field of the skill's SKILL.md frontmatter.
	DisplayName string `json:"displayName,omitempty"`
	// Description describes what the skill does and when to use it. It is one
	// of the fields [SkillRegistryClient.RetrieveSkills] matches against.
	Description string `json:"description,omitempty"`
	// ZippedFilesystem is the zip archive holding SKILL.md at its root plus the
	// optional scripts, references and assets directories. It is only populated
	// by [SkillRegistryClient.GetSkill] and
	// [SkillRegistryClient.GetSkillRevision]; the list methods leave it empty.
	ZippedFilesystem Base64Bytes `json:"zippedFilesystem,omitempty"`
	// License is the SPDX license identifier of the skill, e.g. "Apache-2.0".
	License string `json:"license,omitempty"`
	// Compatibility records the skill's environment requirements.
	Compatibility string `json:"compatibility,omitempty"`
	// Labels holds user-defined metadata.
	Labels map[string]string `json:"labels,omitempty"`
	// State is one of the SkillState* constants. Output only.
	State string `json:"state,omitempty"`
	// SkillSource is one of the SkillSource* constants. Output only.
	SkillSource string `json:"skillSource,omitempty"`
	// SHA256 is the checksum of ZippedFilesystem. Output only.
	SHA256 string `json:"sha256,omitempty"`
	// CreateTime is when the skill was created. Output only.
	CreateTime time.Time `json:"createTime,omitzero"`
	// UpdateTime is when the skill was last updated. Output only.
	UpdateTime time.Time `json:"updateTime,omitzero"`
}

// ID returns the final segment of the skill's resource name — the SKILL_ID the
// skill was created with. It returns "" when Name is empty.
func (s *Skill) ID() string {
	return lastSegment(s.Name)
}

// SkillRevision is one immutable revision of a [Skill].
type SkillRevision struct {
	// Name is the resource name of the revision, in the form
	// "projects/{project}/locations/{location}/skills/{skill}/revisions/{revision}".
	Name string `json:"name,omitempty"`
	// State is one of the SkillState* constants. Output only.
	State string `json:"state,omitempty"`
	// CreateTime is when the revision was created. Output only.
	CreateTime time.Time `json:"createTime,omitzero"`
	// Skill is the state of the skill at this revision. Output only.
	Skill *Skill `json:"skill,omitempty"`
}

// ID returns the final segment of the revision's resource name — the
// REVISION_ID. It returns "" when Name is empty.
func (r *SkillRevision) ID() string {
	return lastSegment(r.Name)
}

// ListSkillsResponse is one page of [SkillRegistryClient.ListSkills] results.
type ListSkillsResponse struct {
	// Skills is the page of skills. Their ZippedFilesystem is not populated.
	Skills []Skill `json:"skills,omitempty"`
	// NextPageToken is the token for the next page, or "" on the last page.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

// ListSkillRevisionsResponse is one page of
// [SkillRegistryClient.ListSkillRevisions] results.
type ListSkillRevisionsResponse struct {
	// SkillRevisions is the page of revisions.
	SkillRevisions []SkillRevision `json:"skillRevisions,omitempty"`
	// NextPageToken is the token for the next page, or "" on the last page.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

// RetrieveSkillsResponse holds the result of a semantic skill search.
type RetrieveSkillsResponse struct {
	// RetrievedSkills are ranked by similarity to the query.
	RetrievedSkills []RetrievedSkill `json:"retrievedSkills,omitempty"`
}

// RetrievedSkill is a single semantic search hit.
type RetrievedSkill struct {
	// SkillName is the resource name of the matched skill; pass it to
	// [SkillRegistryClient.GetSkill] to fetch the full resource.
	SkillName string `json:"skillName,omitempty"`
	// Description is the matched skill's description.
	Description string `json:"description,omitempty"`
}

// ID returns the final segment of the matched skill's resource name.
func (r *RetrievedSkill) ID() string {
	return lastSegment(r.SkillName)
}

// APIError is returned when the Skill Registry responds with a non-2xx status.
// A 404 unwraps to [ErrSkillNotFound], so callers can use
// errors.Is(err, ErrSkillNotFound).
type APIError struct {
	// StatusCode is the HTTP status code of the response.
	StatusCode int
	// Status is the canonical error code from the response body, e.g.
	// "NOT_FOUND". It is empty when the body carries no google.rpc.Status.
	Status string
	// Message is the error message from the response body, if any.
	Message string
	// Body is the raw response body, useful for diagnosing the failure.
	Body string
}

func (e *APIError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("skill registry: API request failed with status %d (%s): %s", e.StatusCode, e.Status, e.Message)
	}
	return fmt.Sprintf("skill registry: API request failed with status %d: %s", e.StatusCode, e.Body)
}

// Unwrap maps a 404 onto the package's [ErrSkillNotFound] sentinel.
func (e *APIError) Unwrap() error {
	if e.StatusCode == http.StatusNotFound {
		return ErrSkillNotFound
	}
	return nil
}

// SkillRegistryClientConfig configures a [SkillRegistryClient].
type SkillRegistryClientConfig struct {
	// ProjectID is the Google Cloud project ID. Required.
	ProjectID string
	// Location is the region hosting the Skill Registry, e.g. "us-central1".
	// Required. It selects the regional endpoint as well as the resource
	// parent, so it must match the region the skills live in.
	Location string
}

// SkillRegistryClient is a read-only REST client for the Skill Registry.
//
// It is safe for concurrent use.
type SkillRegistryClient struct {
	httpClient *http.Client
	// baseURL is the versioned service endpoint, without a trailing slash,
	// e.g. "https://us-central1-aiplatform.googleapis.com/v1beta1".
	baseURL string
	// parent is "projects/{project}/locations/{location}".
	parent string
}

// NewSkillRegistryClient creates a [SkillRegistryClient] for the given project
// and location. By default it authenticates with Application Default
// Credentials against the regional endpoint
// https://{location}-aiplatform.googleapis.com; pass option.WithHTTPClient,
// option.WithEndpoint, option.WithCredentials, etc. to override that. Caller
// options take precedence over the defaults.
func NewSkillRegistryClient(ctx context.Context, cfg SkillRegistryClientConfig, opts ...option.ClientOption) (*SkillRegistryClient, error) {
	if cfg.ProjectID == "" || cfg.Location == "" {
		return nil, errors.New("skill registry: ProjectID and Location must be set")
	}

	// The transport resolves the endpoint, honoring GOOGLE_API_USE_MTLS_ENDPOINT
	// / GOOGLE_API_USE_CLIENT_CERTIFICATE, and returns the one it dialed, so the
	// endpoint and the client certificate can never disagree.
	defaults := []option.ClientOption{
		option.WithScopes(cloudPlatformScope),
		internaloption.WithDefaultEndpoint(regionalEndpoint(cfg.Location, "aiplatform.googleapis.com")),
		internaloption.WithDefaultMTLSEndpoint(regionalEndpoint(cfg.Location, "aiplatform.mtls.googleapis.com")),
	}
	httpClient, endpoint, err := htransport.NewHTTPClient(ctx, append(defaults, opts...)...)
	if err != nil {
		return nil, fmt.Errorf("skill registry: creating HTTP client: %w", err)
	}

	return &SkillRegistryClient{
		httpClient: httpClient,
		baseURL:    strings.TrimSuffix(endpoint, "/"),
		parent:     fmt.Sprintf("projects/%s/locations/%s", cfg.ProjectID, cfg.Location),
	}, nil
}

// ListOption customizes a list request.
type ListOption func(url.Values)

// WithPageSize caps the number of items returned in one page. Values <= 0 are
// ignored, leaving the server default in place.
func WithPageSize(size int) ListOption {
	return func(v url.Values) {
		if size > 0 {
			v.Set("pageSize", strconv.Itoa(size))
		}
	}
}

// WithPageToken requests the page identified by token, as returned in the
// NextPageToken of a previous response. An empty token is ignored.
func WithPageToken(token string) ListOption {
	return func(v url.Values) {
		if token != "" {
			v.Set("pageToken", token)
		}
	}
}

// WithFilter applies an AIP-160 filter. It is only supported by
// [SkillRegistryClient.ListSkillRevisions] (on the "labels" field);
// [SkillRegistryClient.ListSkills] ignores it. An empty filter is ignored.
func WithFilter(filter string) ListOption {
	return func(v url.Values) {
		if filter != "" {
			v.Set("filter", filter)
		}
	}
}

// ListSkills returns one page of the skills registered in the client's project
// and location. The returned skills carry metadata only — use
// [SkillRegistryClient.GetSkill] to fetch a skill's zipped filesystem. For
// automatic paging use [SkillRegistryClient.AllSkills].
func (c *SkillRegistryClient) ListSkills(ctx context.Context, opts ...ListOption) (*ListSkillsResponse, error) {
	var resp ListSkillsResponse
	if err := c.get(ctx, c.parent+"/skills", listValues(opts), &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// AllSkills iterates over every skill in the client's project and location,
// fetching pages on demand. If a page fetch fails the iterator yields a single
// (nil, error) and stops.
func (c *SkillRegistryClient) AllSkills(ctx context.Context, opts ...ListOption) iter.Seq2[*Skill, error] {
	return pages(ctx, opts, func(ctx context.Context, o ...ListOption) ([]Skill, string, error) {
		page, err := c.ListSkills(ctx, o...)
		if err != nil {
			return nil, "", err
		}
		return page.Skills, page.NextPageToken, nil
	})
}

// GetSkill returns the latest revision of a single skill, including its zipped
// filesystem. nameOrID is either a bare skill ID ("my-skill") or a full
// resource name ("projects/{project}/locations/{location}/skills/my-skill").
func (c *SkillRegistryClient) GetSkill(ctx context.Context, nameOrID string) (*Skill, error) {
	name, err := c.skillName(nameOrID)
	if err != nil {
		return nil, err
	}
	var sk Skill
	if err := c.get(ctx, name, nil, &sk); err != nil {
		return nil, err
	}
	return &sk, nil
}

// RetrieveSkills finds skills by semantic search over their display names and
// descriptions, for example "skills to manage cloud resources". topK caps the
// number of hits (max 100); values <= 0 leave the server default of 10 in
// place. The hits carry no payload — pass RetrievedSkill.SkillName to
// [SkillRegistryClient.GetSkill] to fetch one.
func (c *SkillRegistryClient) RetrieveSkills(ctx context.Context, query string, topK int) (*RetrieveSkillsResponse, error) {
	params := url.Values{}
	if query != "" {
		params.Set("query", query)
	}
	if topK > 0 {
		params.Set("topK", strconv.Itoa(topK))
	}
	var resp RetrieveSkillsResponse
	if err := c.get(ctx, c.parent+"/skills:retrieve", params, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ListSkillRevisions returns one page of the revision history of a skill,
// newest first. nameOrID is a bare skill ID or a full skill resource name. For
// automatic paging use [SkillRegistryClient.AllSkillRevisions].
func (c *SkillRegistryClient) ListSkillRevisions(ctx context.Context, nameOrID string, opts ...ListOption) (*ListSkillRevisionsResponse, error) {
	name, err := c.skillName(nameOrID)
	if err != nil {
		return nil, err
	}
	var resp ListSkillRevisionsResponse
	if err := c.get(ctx, name+"/revisions", listValues(opts), &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// AllSkillRevisions iterates over every revision of a skill, fetching pages on
// demand. If a page fetch fails the iterator yields a single (nil, error) and
// stops.
func (c *SkillRegistryClient) AllSkillRevisions(ctx context.Context, nameOrID string, opts ...ListOption) iter.Seq2[*SkillRevision, error] {
	return pages(ctx, opts, func(ctx context.Context, o ...ListOption) ([]SkillRevision, string, error) {
		page, err := c.ListSkillRevisions(ctx, nameOrID, o...)
		if err != nil {
			return nil, "", err
		}
		return page.SkillRevisions, page.NextPageToken, nil
	})
}

// GetSkillRevision returns one revision of a skill, including the state of the
// skill at that revision. nameOrID is a bare skill ID or a full skill resource
// name; revisionID is the final segment of a revision resource name, as
// returned by [SkillRegistryClient.ListSkillRevisions].
func (c *SkillRegistryClient) GetSkillRevision(ctx context.Context, nameOrID, revisionID string) (*SkillRevision, error) {
	name, err := c.skillName(nameOrID)
	if err != nil {
		return nil, err
	}
	if revisionID == "" || strings.Contains(revisionID, "/") {
		return nil, fmt.Errorf("skill registry: %w: revision ID %q must be a single path segment", ErrInvalidSkillName, revisionID)
	}
	var rev SkillRevision
	if err := c.get(ctx, name+"/revisions/"+revisionID, nil, &rev); err != nil {
		return nil, err
	}
	return &rev, nil
}

// skillName expands a bare skill ID into a full resource name and validates a
// name that is already fully qualified.
func (c *SkillRegistryClient) skillName(nameOrID string) (string, error) {
	if strings.HasPrefix(nameOrID, "projects/") {
		// projects/{p}/locations/{l}/skills/{skill}
		if parts := strings.Split(nameOrID, "/"); len(parts) == 6 && parts[2] == "locations" && parts[4] == "skills" && parts[5] != "" {
			return nameOrID, nil
		}
		return "", fmt.Errorf("skill registry: %w: %q is not of the form projects/{project}/locations/{location}/skills/{skill}", ErrInvalidSkillName, nameOrID)
	}
	if nameOrID == "" || strings.Contains(nameOrID, "/") {
		return "", fmt.Errorf("skill registry: %w: %q must be a skill ID or a full resource name", ErrInvalidSkillName, nameOrID)
	}
	return c.parent + "/skills/" + nameOrID, nil
}

// get issues an authenticated GET for resourcePath — a resource name relative
// to the service endpoint — applies params as the query string, and decodes the
// JSON response body into v. A non-2xx response is returned as an [*APIError].
func (c *SkillRegistryClient) get(ctx context.Context, resourcePath string, params url.Values, v any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/"+resourcePath, nil)
	if err != nil {
		return fmt.Errorf("skill registry: building request: %w", err)
	}
	if len(params) > 0 {
		req.URL.RawQuery = params.Encode()
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("skill registry: GET %s: %w", resourcePath, err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("skill registry: reading response body: %w", err)
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return newAPIError(resp.StatusCode, body)
	}
	if v != nil {
		if err := json.Unmarshal(body, v); err != nil {
			return fmt.Errorf("skill registry: decoding response: %w", err)
		}
	}
	return nil
}

// newAPIError builds an [APIError], extracting the canonical status and message
// from a google.rpc.Status body when the response carries one.
func newAPIError(statusCode int, body []byte) *APIError {
	e := &APIError{StatusCode: statusCode, Body: string(body)}
	var parsed struct {
		Error struct {
			Status  string `json:"status"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &parsed); err == nil {
		e.Status = parsed.Error.Status
		e.Message = parsed.Error.Message
	}
	return e
}

// listValues folds opts into the query parameters of a list request.
func listValues(opts []ListOption) url.Values {
	v := url.Values{}
	for _, o := range opts {
		o(v)
	}
	return v
}

// pages returns an iterator that yields every item across all pages, fetching
// subsequent pages on demand via fetch. fetch returns one page of items and the
// next page token (empty when there are no more pages). On error the iterator
// yields a single (nil, err) and stops.
func pages[T any](
	ctx context.Context,
	opts []ListOption,
	fetch func(context.Context, ...ListOption) (items []T, nextPageToken string, err error),
) iter.Seq2[*T, error] {
	return func(yield func(*T, error) bool) {
		token := ""
		for {
			pageOpts := opts
			if token != "" {
				// Clone so we never append into opts' backing array, which the
				// caller owns and we reuse on every iteration.
				pageOpts = append(slices.Clone(opts), WithPageToken(token))
			}

			items, next, err := fetch(ctx, pageOpts...)
			if err != nil {
				yield(nil, err)
				return
			}
			for i := range items {
				if !yield(&items[i], nil) {
					return
				}
			}
			if next == "" {
				return
			}
			token = next
		}
	}
}

// regionalEndpoint builds the versioned endpoint for a regional Agent Platform
// service host, e.g. "https://us-central1-aiplatform.googleapis.com/v1beta1".
func regionalEndpoint(location, host string) string {
	return fmt.Sprintf("https://%s-%s/%s", location, host, skillRegistryAPIVersion)
}

// lastSegment returns the substring after the final "/" of name.
func lastSegment(name string) string {
	if i := strings.LastIndex(name, "/"); i >= 0 {
		return name[i+1:]
	}
	return name
}

// Base64Bytes is binary data that the JSON API transports as a base64 string.
//
// Unmarshaling accepts both the standard and the URL-safe alphabet, padded or
// unpadded, as permitted by the proto3 JSON mapping; marshaling always emits
// padded standard base64.
type Base64Bytes []byte

// MarshalJSON implements [json.Marshaler].
func (b Base64Bytes) MarshalJSON() ([]byte, error) {
	if b == nil {
		return []byte("null"), nil
	}
	return json.Marshal(base64.StdEncoding.EncodeToString(b))
}

// UnmarshalJSON implements [json.Unmarshaler].
func (b *Base64Bytes) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return fmt.Errorf("skill registry: base64 field is not a JSON string: %w", err)
	}
	if s == "" {
		*b = nil
		return nil
	}
	enc := base64.StdEncoding
	if strings.ContainsAny(s, "-_") {
		enc = base64.URLEncoding
	}
	if len(s)%4 != 0 {
		enc = enc.WithPadding(base64.NoPadding)
	}
	decoded, err := enc.DecodeString(s)
	if err != nil {
		return fmt.Errorf("skill registry: decoding base64 field: %w", err)
	}
	*b = decoded
	return nil
}
