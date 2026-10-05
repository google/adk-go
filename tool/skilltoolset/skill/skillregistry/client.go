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
	"context"
	"fmt"
	"io"
	"log"
	"regexp"
	"strings"

	agentregistry "google.golang.org/api/agentregistry/v1alpha"
)

// Config allowes you to indicate the resources your [client] should be providing
type Config struct {
	ProjectID             string
	Location              string
	IncludeUnusableSkills bool
}

// Client defines the interface for a Client accessing SkillRegistry
type Client interface {
	ListSkills() ([]*agentregistry.Skill, error)
	ListFrontmatters() ([]*agentregistry.Frontmatter, error)
	GetSkill(name string) (*agentregistry.Skill, error)
	GetRevision(rev string) (*agentregistry.SkillRevision, error)
	GetZip(rev string) (*zip.Reader, error)
	FindFrontmatters(query string) ([]*agentregistry.Frontmatter, error)
	ResourceID(name string) string
	ParseSkillName(name string) (projectID, location, skillName string, resErr error)
}

type client struct {
	parent                string
	namePrefix            string
	pageSize              int64
	svc                   *agentregistry.ProjectsLocationsSkillsService
	includeUnusableSkills bool
}

// NewClient returns a new client using SkillRegistry as a backend
func NewClient(ctx context.Context, cfg Config) (Client, error) {
	as, err := agentregistry.NewService(ctx)
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

// iterateDo iterates over the skills using acc to accumulated the data. acc can return false to stop the iteration
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

	r := strings.NewReader(string(b))
	zip, err := zip.NewReader(r, int64(len(b)))
	if err != nil {
		return nil, fmt.Errorf("cannot read the zip: %w", err)
	}
	for _, f := range zip.File {
		log.Printf("File: %v", f.Name)
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
			// log.Printf("FindFrontmatters got skill with non-nil frontmatter; skill: %+v frontmatter: %+v", sk, fm)
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

	resp, err := call.Download()
	if err != nil {
		return nil, fmt.Errorf("cannot download the revision: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("cannot read the revision: %w", err)
	}
	log.Printf("Read %v bytes", len(b))

	r := strings.NewReader(string(b))
	zip, err := zip.NewReader(r, int64(len(b)))
	if err != nil {
		return nil, fmt.Errorf("cannot read the zip: %w", err)
	}
	for _, f := range zip.File {
		log.Printf("File: %v", f.Name)
	}

	return sr, nil
}

func (c *client) FindSkills(searchType, query string) ([]*agentregistry.Frontmatter, error) {
	res := make([]*agentregistry.Frontmatter, 0)

	s := c.svc.Search(c.parent)
	s.SearchType(searchType)
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
		cont := true
		for _, sk := range resp.Skills {
			res = append(res, sk.Frontmatter)

			log.Printf("FindSkills got skill: %+v", sk)
		}
		if !cont { // acc returned false in order no to continue
			break
		}

		if pageToken == "" {
			break
		}
	}
	return res, nil
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

		// b, err := f.MarshalJSON()
		// if err != nil {
		// 	log.Printf("cannot MarshalJSON for f")
		// 	return false, nil
		// }

		// log.Printf("ListFrontmatters: GOT SKILL: %+v", string(b))

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
	// fullname := fmt.Sprintf("%v/skills/%v", c.parent, name)
	call := c.svc.Get(name)
	s, err := call.Do()
	if err != nil {
		return nil, fmt.Errorf("cannot Get: Do failed: %w", err)
	}
	if s == nil {
		return nil, fmt.Errorf("got a skill but it is nil")
	}
	// log.Printf("got a skill : %+v", s)
	return s, nil
}

// // ParseSkillID parses strings according to skillIDRegex.
// // Returns an error if the skillID doesn't match.
// // Example of a valid skillID: `urn:skill:projects-1234567890:locations:global:private-skill-name`
// func (c *client) ParseSkillID(skillID string) (projectID, location, skillName string, resErr error) {

// 	matches := skillIDRegex.FindStringSubmatch(skillID)
// 	if len(matches) != 4 {
// 		return "", "", "", fmt.Errorf("invalid skillID format")
// 	}
// 	projectID = matches[1]
// 	location = matches[2]
// 	skillName = matches[3]

// 	return projectID, location, skillName, nil
// }

// Exampple: `projects/your-project-name/locations/global/skills/cloud.google.com-google-cloud-solution-n-tier-serverless-web-app`

var skillIDRegex = regexp.MustCompile("projects/([^/]+)/locations/([^/]+)/skills/([^/]+)")

// ParseSkillID parses strings according to skillIDRegex.
// Returns an error if the skillID doesn't match.
// Example of a valid skillID: `urn:skill:projects-1234567890:locations:global:private-skill-name`
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
