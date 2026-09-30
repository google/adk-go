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
	"google.golang.org/adk/v2/tool/skilltoolset/skill"
	agentregistry "google.golang.org/api/agentregistry/v1alpha"
)

type cachedSkill struct {
	client Client

	origSkill   agentregistry.Skill
	frontmatter skill.Frontmatter

	resources *resources
}

// newCachedSkill returns a new cachedSkill containting the provided agentregistry.Skill
func newCachedSkill(client Client, s *agentregistry.Skill) *cachedSkill {
	return &cachedSkill{
		client:    client,
		origSkill: *s,
		resources: newResources(client),
	}
}

func (s *cachedSkill) getResource(path string) (*resource, error) {
	return s.resources.getResource(s.origSkill.DefaultRevision, path)
}

func (s *cachedSkill) listResources(subpath string) ([]string, error) {
	return s.resources.listResources(s.origSkill.DefaultRevision, subpath)
}
