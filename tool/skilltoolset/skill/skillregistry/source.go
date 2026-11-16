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
	"context"
	"fmt"
	"io"

	agentregistry "google.golang.org/api/agentregistry/v1alpha"
	"google.golang.org/api/option"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
)

// SkillRegistrySourceConfig configures a [skill.Source] backed by the Skill Registry.
type SkillRegistrySourceConfig struct {
	// ProjectID is the Google Cloud project ID. Required unless Client is set.
	ProjectID string
	// Location is the region hosting the Skill Registry, e.g. "us-central1".
	// Required unless Client is set.
	Location string
	// Client, when non-nil, is used as-is; ProjectID and Location are then
	// ignored, as are any option.ClientOption passed to
	// [NewSkillRegistrySource].
	Client Client
}

// NewSkillRegistrySource creates a [skill.Source] backed by the Skill Registry
// of the Gemini Enterprise Agent Platform.
//
// Skills are keyed by their SKILL_ID — the last segment of the skill's resource
// name — because that is the folder name the skill is mounted under when it is
// attached to an agent. [skill.Frontmatter] is built from the registered skill
// metadata rather than from the SKILL.md inside the payload, so that listing
// and loading a skill can never disagree; the SKILL.md "metadata" and
// "allowed-tools" fields are therefore not surfaced, as the API does not model
// them.
//
// Every method issues at least one API call, and everything except
// ListFrontmatters downloads the skill's whole zipped filesystem. Wrap the
// source in a cache from the simplecache subpackage
// (simplecache.NewCachedSkillRegistrySource) to serve repeated reads from
// memory.
func NewSkillRegistrySource(ctx context.Context, cfg SkillRegistrySourceConfig, opts ...option.ClientOption) (*SkillRegistrySource, error) {
	client := cfg.Client
	if client == nil {
		c, err := NewClient(ctx, Config{
			ProjectID: cfg.ProjectID,
			Location:  cfg.Location,
		})
		if err != nil {
			return nil, err
		}
		client = c
	}
	return &SkillRegistrySource{client: client}, nil
}

// SkillRegistrySource serves skills straight from the Skill Registry. It holds
// no state beyond its client, so it is safe for concurrent use.
type SkillRegistrySource struct {
	client Client
}

// ListFrontmatters implements [skill.Source].
func (s *SkillRegistrySource) ListFrontmatters(ctx context.Context) ([]*skill.Frontmatter, error) {
	lf, err := s.client.ListFrontmatters()
	if err != nil {
		return nil, fmt.Errorf("cannot list frontmatters: %w", err)
	}

	res := make([]*skill.Frontmatter, 0)
	for _, f := range lf {
		sf := s.toFrontmatter(f)
		if sf == nil {
			continue
		}
		res = append(res, sf)
	}
	return res, nil
}

func (s *SkillRegistrySource) toFrontmatter(f *agentregistry.Frontmatter) *skill.Frontmatter {
	if f == nil {
		return nil
	}
	return &skill.Frontmatter{
		Name:          s.client.ResourceID(f.Name),
		Description:   f.Description,
		License:       f.License,
		Compatibility: f.Compatibility,
		Metadata:      f.Metadata,
	}
}

// ListResources implements [skill.Source].
func (s *SkillRegistrySource) ListResources(ctx context.Context, name, subpath string) ([]string, error) {
	res := make([]string, 0)
	zip, err := s.client.GetZip(name)
	if err != nil {
		return nil, fmt.Errorf("cannot GetZip: %w", err)
	}

	for _, f := range zip.File {
		res = append(res, f.Name)
	}
	return res, nil
}

// LoadFrontmatter implements [skill.Source].
func (s *SkillRegistrySource) LoadFrontmatter(ctx context.Context, name string) (*skill.Frontmatter, error) {
	skill, err := s.client.GetSkill(name)
	if err != nil {
		return nil, fmt.Errorf("cannot GetSkill: %w", err)
	}
	if skill == nil {
		return nil, fmt.Errorf("got nil from GetSkill")
	}
	sf := s.toFrontmatter(skill.Frontmatter)

	return sf, nil
}

// LoadInstructions implements [skill.Source].
func (s *SkillRegistrySource) LoadInstructions(ctx context.Context, name string) (string, error) {
	sk, err := s.client.GetSkill(name)
	if err != nil {
		return "", fmt.Errorf("cannot GetSkill: %w", err)
	}
	if sk == nil {
		return "", fmt.Errorf("got nil from GetSkill")
	}
	r, err := s.client.GetZip(sk.DefaultRevision)
	if err != nil {
		return "", fmt.Errorf("cannot GetZip: %w", err)
	}
	// find SKILL.md
	for _, f := range r.File {
		if f.Name == "SKILL.md" {
			b, err := f.Open()
			if err != nil {
				return "", fmt.Errorf("cannot open SKILL.md: %w", err)
			}
			defer func() { _ = b.Close() }()
			content, err := io.ReadAll(b)
			if err != nil {
				return "", fmt.Errorf("cannot read SKILL.md: %w", err)
			}
			_, instr, err := skill.ParseBytes(content)
			if err != nil {
				return "", fmt.Errorf("cannot parse SKILL.md: %w", err)
			}

			return instr, nil
		}
	}

	return sk.Frontmatter.Description, nil
}

// LoadResource implements [skill.Source].
func (s *SkillRegistrySource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	zip, err := s.client.GetZip(name)
	if err != nil {
		return nil, fmt.Errorf("cannot GetZip: %w", err)
	}

	for _, f := range zip.File {
		if f.Name == resourcePath {
			return f.Open()
		}
	}
	return nil, fmt.Errorf("cannot find the requested resource %v", resourcePath)
}

var _ skill.Source = (*SkillRegistrySource)(nil)
