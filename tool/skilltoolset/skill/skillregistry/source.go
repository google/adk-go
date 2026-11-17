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
	"strings"

	agentregistry "google.golang.org/api/agentregistry/v1alpha"
	"google.golang.org/api/option"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/tool/skilltoolset/skill"
)

// safeResourcePath validates p as a path inside a skill's in-memory filesystem
// and returns its cleaned, slash-separated form. It returns
// [skill.ErrInvalidResourcePath] for an absolute path, or one that escapes the
// skill root through "..".
func safeResourcePath(p string) (string, error) {
	clean, err := utils.SafeRelPath(p)
	if err != nil {
		return "", fmt.Errorf("%w: %q", skill.ErrInvalidResourcePath, p)
	}
	return clean, nil
}

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
		}, opts...)
		if err != nil {
			return nil, err
		}
		client = c
	}
	return &SkillRegistrySource{client: client}, nil
}

// defaultRevision resolves a skill's full resource name to the revision its
// archive is downloaded from.
func (s *SkillRegistrySource) defaultRevision(name string) (string, error) {
	sk, err := s.client.GetSkill(name)
	if err != nil {
		return "", fmt.Errorf("cannot GetSkill: %w", err)
	}
	if sk == nil {
		return "", fmt.Errorf("got nil from GetSkill")
	}
	if sk.DefaultRevision == "" {
		return "", fmt.Errorf("skill has no default revision")
	}
	return sk.DefaultRevision, nil
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

// ListResources implements [skill.Source]. An empty subpath, ".", "/" or "\"
// lists every file; otherwise subpath names a directory (or a single file) and
// only files at or below it are returned. A traversal or absolute subpath
// returns [skill.ErrInvalidResourcePath].
func (s *SkillRegistrySource) ListResources(ctx context.Context, name, subpath string) ([]string, error) {
	isRoot := subpath == "" || subpath == "." || subpath == "/" || subpath == "\\"
	prefix := ""
	if !isRoot {
		clean, err := safeResourcePath(subpath)
		if err != nil {
			return nil, err
		}
		prefix = clean
	}

	rev, err := s.defaultRevision(name)
	if err != nil {
		return nil, err
	}
	zr, err := s.client.GetZip(rev)
	if err != nil {
		return nil, fmt.Errorf("cannot GetZip: %w", err)
	}

	res := make([]string, 0)
	for _, f := range zr.File {
		if strings.HasSuffix(f.Name, "/") {
			continue // directory entry
		}
		// The archive is untrusted; skip any entry that escapes the skill root.
		clean, err := safeResourcePath(f.Name)
		if err != nil {
			continue
		}
		if isRoot || clean == prefix || strings.HasPrefix(clean, prefix+"/") {
			res = append(res, clean)
		}
	}
	return res, nil
}

// LoadFrontmatter implements [skill.Source].
func (s *SkillRegistrySource) LoadFrontmatter(ctx context.Context, name string) (*skill.Frontmatter, error) {
	sk, err := s.client.GetSkill(name)
	if err != nil {
		return nil, fmt.Errorf("cannot GetSkill: %w", err)
	}
	if sk == nil {
		return nil, fmt.Errorf("got nil from GetSkill")
	}
	if sk.Frontmatter == nil {
		return nil, fmt.Errorf("skill has no frontmatter")
	}
	return s.toFrontmatter(sk.Frontmatter), nil
}

// LoadInstructions implements [skill.Source]. It returns the Markdown body of
// the skill's SKILL.md, or [skill.ErrResourceNotFound] if the archive has none.
func (s *SkillRegistrySource) LoadInstructions(ctx context.Context, name string) (string, error) {
	rev, err := s.defaultRevision(name)
	if err != nil {
		return "", err
	}
	r, err := s.client.GetZip(rev)
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

	return "", fmt.Errorf("%w: SKILL.md", skill.ErrResourceNotFound)
}

// LoadResource implements [skill.Source]. It returns
// [skill.ErrResourceNotFound] when the skill has no such file, and
// [skill.ErrInvalidResourcePath] for a traversal or absolute resourcePath.
func (s *SkillRegistrySource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	clean, err := safeResourcePath(resourcePath)
	if err != nil {
		return nil, err
	}

	rev, err := s.defaultRevision(name)
	if err != nil {
		return nil, err
	}
	zr, err := s.client.GetZip(rev)
	if err != nil {
		return nil, fmt.Errorf("cannot GetZip: %w", err)
	}

	for _, f := range zr.File {
		if strings.HasSuffix(f.Name, "/") {
			continue // directory entry
		}
		fc, err := safeResourcePath(f.Name)
		if err != nil {
			continue
		}
		if fc == clean {
			return f.Open()
		}
	}
	return nil, fmt.Errorf("%w: %q", skill.ErrResourceNotFound, clean)
}

var _ skill.Source = (*SkillRegistrySource)(nil)
