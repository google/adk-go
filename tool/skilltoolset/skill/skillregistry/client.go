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

package skillregistry

import (
	"archive/zip"
	"bytes"
	"context"
	"fmt"
	"io"
	"regexp"

	agentregistry "google.golang.org/api/agentregistry/v1alpha"
	"google.golang.org/api/option"
)

// Config holds the parameters for [NewClient].
type Config struct {
	// ProjectID is the Google Cloud project that owns the Skill Registry.
	ProjectID string
	// Location is the region hosting the Skill Registry, e.g. "us-central1".
	Location string
	// IncludeUnusableSkills, when true, also returns skills in a state that
	// cannot be read (anything other than active or deprecated). Defaults to
	// false.
	IncludeUnusableSkills bool
}

// Client is a low-level client for the Skill Registry. Its methods map closely
// onto the underlying API; most callers want a skill.Source from
// [NewSkillRegistrySource] instead.
type Client interface {
	// ListSkills returns every skill in the configured project and location,
	// following pagination. Skills in an unusable state are omitted unless
	// Config.IncludeUnusableSkills was set.
	ListSkills() ([]*agentregistry.Skill, error)
	// ListFrontmatters returns the frontmatter of every listed skill, fetching
	// each skill individually.
	ListFrontmatters() ([]*agentregistry.Frontmatter, error)
	// GetSkill returns the skill identified by its full resource name.
	GetSkill(name string) (*agentregistry.Skill, error)
	// GetRevision returns the metadata of the named skill revision.
	GetRevision(rev string) (*agentregistry.SkillRevision, error)
	// GetZip downloads the named revision and returns a reader over its zipped
	// filesystem.
	GetZip(rev string) (*zip.Reader, error)
	// FindFrontmatters returns the frontmatter of skills matching a keyword
	// search.
	FindFrontmatters(query string) ([]*agentregistry.Frontmatter, error)
	// ResourceID returns the full resource name
	// ("projects/<p>/locations/<l>/skills/<name>") for a skill's short name.
	ResourceID(name string) string
	// ParseSkillName splits a full skill resource name into its project,
	// location and skill-name parts, returning an error if it is malformed.
	ParseSkillName(name string) (projectID, location, skillName string, resErr error)
}

type client struct {
	parent                string
	namePrefix            string
	pageSize              int64
	svc                   *agentregistry.ProjectsLocationsSkillsService
	includeUnusableSkills bool
}

// NewClient returns a [Client] backed by the Skill Registry for the project and
// location in cfg. The option.ClientOption values are forwarded to the Agent
// Registry service, e.g. to supply credentials or an alternate endpoint. It
// fails if the service cannot be initialized, for example when credentials are
// unavailable.
func NewClient(ctx context.Context, cfg Config, opts ...option.ClientOption) (Client, error) {
	as, err := agentregistry.NewService(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("agentregistry.NewService failed: %w", err)
	}

	svc := agentregistry.NewProjectsLocationsSkillsService(as)
	return &client{
		svc:                   svc,
		pageSize:              40,
		parent:                fmt.Sprintf(`projects/%v/locations/%v`, cfg.ProjectID, cfg.Location),
		namePrefix:            fmt.Sprintf(`projects/%v/locations/`, cfg.ProjectID),
		includeUnusableSkills: cfg.IncludeUnusableSkills,
	}, nil
}

func (c *client) ResourceID(name string) string {
	return fmt.Sprintf("%v/skills/%v", c.parent, name)
}

// isUsableSkill reports whether a skill in the given state can be read.
func isUsableSkill(state string) bool {
	return state == "STATE_ACTIVE" || state == "STATE_DEPRECATED"
}

// iterateDo iterates over the skills, passing each to acc to accumulate data.
// acc can return false to stop the iteration.
func (c *client) iterateDo(acc func(*agentregistry.Skill) (bool, error)) error {
	lc := c.svc.List(c.parent)
	lc.PageSize(c.pageSize)
	pageToken := ""

	for {
		lc.PageToken(pageToken)
		resp, err := lc.Do()
		if err != nil {
			return fmt.Errorf("cannot List skills: %w", err)
		}
		pageToken = resp.NextPageToken
		cont := true
		for _, sk := range resp.Skills {
			if !c.includeUnusableSkills && !isUsableSkill(sk.State) {
				continue
			}
			cont, err = acc(sk)
			if err != nil {
				return fmt.Errorf("acc failed: %w", err)
			}
			if !cont {
				break
			}
		}
		if !cont { // acc returned false in order no to continue
			break
		}

		if pageToken == "" {
			break
		}
	}
	return nil
}

func (c *client) GetZip(rev string) (*zip.Reader, error) {
	call := c.svc.Revisions.Get(rev)

	resp, err := call.Download()
	if err != nil {
		return nil, fmt.Errorf("cannot download the revision: %w", err)
	}
	// ignore the error on Close
	defer func() { _ = resp.Body.Close() }()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("cannot read the revision: %w", err)
	}

	zip, err := zip.NewReader(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		return nil, fmt.Errorf("cannot read the zip: %w", err)
	}

	return zip, nil
}

func (c *client) FindFrontmatters(query string) ([]*agentregistry.Frontmatter, error) {
	res := make([]*agentregistry.Frontmatter, 0)

	s := c.svc.Search(c.parent)
	s.SearchType("KEYWORD")
	s.SearchString(query)
	s.PageSize(c.pageSize)
	pageToken := ""
	for {
		s.PageToken(pageToken)
		resp, err := s.Do()
		if err != nil {
			return nil, fmt.Errorf("cannot List skills: %w", err)
		}
		pageToken = resp.NextPageToken
		for _, sk := range resp.Skills {
			fm := sk.Frontmatter
			if fm == nil {
				f, err := c.GetSkill(sk.Name)
				if err != nil {
					return nil, fmt.Errorf("cannot get the skill: %w", err)
				}
				fm = f.Frontmatter
			}
			if fm == nil {
				// no frontmatter
				continue
			}
			res = append(res, fm)
		}
		if pageToken == "" {
			break
		}
	}
	return res, nil
}

func (c *client) GetRevision(rev string) (*agentregistry.SkillRevision, error) {
	call := c.svc.Revisions.Get(rev)
	sr, err := call.Do()
	if err != nil {
		return nil, fmt.Errorf("cannot get the revision: %w", err)
	}
	return sr, nil
}

func (c *client) ListFrontmatters() ([]*agentregistry.Frontmatter, error) {
	res := make([]*agentregistry.Frontmatter, 0)

	err := c.iterateDo(func(s *agentregistry.Skill) (bool, error) {
		if s == nil {
			return false, fmt.Errorf("skill cannot be nil")
		}

		f, err := c.GetSkill(s.Name)
		if err != nil {
			return false, fmt.Errorf("cannot get the skill: %w", err)
		}

		if f == nil {
			return false, fmt.Errorf("skill cannot be nil")
		}

		res = append(res, f.Frontmatter)

		return true, nil
	})
	if err != nil {
		return nil, fmt.Errorf("cannot iterate over the skills: %w", err)
	}

	return res, nil
}

func (c *client) ListSkills() ([]*agentregistry.Skill, error) {
	res := make([]*agentregistry.Skill, 0)

	err := c.iterateDo(func(s *agentregistry.Skill) (bool, error) {
		res = append(res, s)
		return true, nil
	})
	if err != nil {
		return nil, fmt.Errorf("cannot iterate over the skills: %w", err)
	}

	return res, nil
}

func (c *client) GetSkill(name string) (*agentregistry.Skill, error) {
	call := c.svc.Get(name)
	s, err := call.Do()
	if err != nil {
		return nil, fmt.Errorf("cannot Get: Do failed: %w", err)
	}
	if s == nil {
		return nil, fmt.Errorf("got a skill but it is nil")
	}
	return s, nil
}

// skillIDRegex matches a full skill resource name and captures its project,
// location and skill-name parts. For example:
// projects/your-project/locations/global/skills/your.provider-your-skill-name
var skillIDRegex = regexp.MustCompile("projects/([^/]+)/locations/([^/]+)/skills/([^/]+)")

// ParseSkillName splits a full skill resource name
// ("projects/<p>/locations/<l>/skills/<name>") into its project, location and
// skill-name parts, returning an error if it is malformed.
func (c *client) ParseSkillName(name string) (projectID, location, skillName string, resErr error) {
	matches := skillIDRegex.FindStringSubmatch(name)
	if len(matches) != 4 {
		return "", "", "", fmt.Errorf("invalid skillID format")
	}
	projectID = matches[1]
	location = matches[2]
	skillName = matches[3]

	return projectID, location, skillName, nil
}
