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
	"context"
	"errors"
	"io"
)

// Common errors returned by Source implementations.
var (
	ErrInvalidSkillName    = errors.New("invalid skill name")
	ErrInvalidFrontmatter  = errors.New("invalid frontmatter")
	ErrSkillNotFound       = errors.New("skill not found")
	ErrDuplicateSkill      = errors.New("duplicate skill")
	ErrInvalidResourcePath = errors.New("invalid resource path")
	ErrResourceNotFound    = errors.New("resource not found")
)

// Source is the interface for accessing skill components.
//
// Implementations must:
//   - be safe for concurrent use.
//   - return the sentinel error values defined in this package
//     (ErrInvalidSkillName, ErrInvalidFrontmatter, ErrSkillNotFound,
//     ErrDuplicateSkill, ErrInvalidResourcePath, ErrResourceNotFound, etc.)
//     when appropriate.
type Source interface {
	// ListFrontmatters returns frontmatters for all available skills.
	ListFrontmatters(ctx context.Context) ([]*Frontmatter, error)

	// ListResources returns resource paths for a given skill and subpath.
	// subpath is a relative path to the skill, as are the returned resource
	// paths.
	ListResources(ctx context.Context, name, subpath string) ([]string, error)

	// LoadFrontmatter returns the frontmatter for a single skill by name.
	LoadFrontmatter(ctx context.Context, name string) (*Frontmatter, error)

	// LoadInstructions returns the instruction body for a single skill by name.
	LoadInstructions(ctx context.Context, name string) (string, error)

	// LoadResource returns a stream for a specific resource within a given skill.
	// resourcePath is a relative path to the skill.
	//
	// NOTE: The caller is responsible for closing the returned io.ReadCloser.
	LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error)
}
