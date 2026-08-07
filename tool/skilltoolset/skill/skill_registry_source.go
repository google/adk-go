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
	"archive/zip"
	"bytes"
	"context"
	"fmt"
	"io"
	"path"
	"slices"
	"strings"

	"google.golang.org/api/option"
)

// skillResourceDirs are the top-level directories of a skill's filesystem that
// hold addressable resources. See
// https://agentskills.io/specification#directory-structure.
var skillResourceDirs = []string{"assets", "references", "scripts"}

// SkillRegistrySourceConfig configures a [Source] backed by the Skill Registry.
type SkillRegistrySourceConfig struct {
	// ProjectID is the Google Cloud project ID. Required unless Client is set.
	ProjectID string
	// Location is the region hosting the Skill Registry, e.g. "us-central1".
	// Required unless Client is set.
	Location string
	// Client, when non-nil, is used as-is; ProjectID and Location are then
	// ignored, as are any option.ClientOption passed to
	// [NewSkillRegistrySource].
	Client *SkillRegistryClient
}

// NewSkillRegistrySource creates a [Source] backed by the Skill Registry of the
// Gemini Enterprise Agent Platform. Client options are forwarded to
// [NewSkillRegistryClient].
//
// Skills are keyed by their SKILL_ID — the last segment of the skill's resource
// name — because that is the folder name the skill is mounted under when it is
// attached to an agent. [Frontmatter] is built from the registered skill
// metadata rather than from the SKILL.md inside the payload, so that listing
// and loading a skill can never disagree; the SKILL.md "metadata" and
// "allowed-tools" fields are therefore not surfaced, as the API does not model
// them.
//
// Every method issues at least one API call, and everything except
// ListFrontmatters downloads the skill's whole zipped filesystem. Wrap the
// source in [WithCompletePreloadSource] (or [WithFrontmatterPreloadSource]) to
// serve repeated reads from memory.
func NewSkillRegistrySource(ctx context.Context, cfg SkillRegistrySourceConfig, opts ...option.ClientOption) (Source, error) {
	client := cfg.Client
	if client == nil {
		c, err := NewSkillRegistryClient(ctx, SkillRegistryClientConfig{
			ProjectID: cfg.ProjectID,
			Location:  cfg.Location,
		}, opts...)
		if err != nil {
			return nil, err
		}
		client = c
	}
	return &skillRegistrySource{client: client}, nil
}

// skillRegistrySource serves skills straight from the Skill Registry. It holds
// no state beyond its client, so it is safe for concurrent use.
type skillRegistrySource struct {
	client *SkillRegistryClient
}

var _ Source = (*skillRegistrySource)(nil)

// ListFrontmatters returns the frontmatter of every skill registered in the
// client's project and location, sorted by name. Skills that are still being
// created, or that failed to process or are being deleted, are skipped. It
// reads registered metadata only and does not download any payload.
func (s *skillRegistrySource) ListFrontmatters(ctx context.Context) ([]*Frontmatter, error) {
	var frontmatters []*Frontmatter
	for sk, err := range s.client.AllSkills(ctx) {
		if err != nil {
			return nil, fmt.Errorf("list skills: %w", err)
		}
		if !isUsableSkill(sk.State) {
			continue
		}
		frontmatters = append(frontmatters, frontmatterOf(sk))
	}
	slices.SortFunc(frontmatters, func(a, b *Frontmatter) int {
		return strings.Compare(a.Name, b.Name)
	})
	return frontmatters, nil
}

// LoadFrontmatter returns the frontmatter of a single skill.
func (s *skillRegistrySource) LoadFrontmatter(ctx context.Context, name string) (*Frontmatter, error) {
	sk, err := s.client.GetSkill(ctx, name)
	if err != nil {
		return nil, fmt.Errorf("get skill %q: %w", name, err)
	}
	return frontmatterOf(sk), nil
}

// LoadInstructions returns the Markdown body of the skill's SKILL.md, that is
// everything after the YAML frontmatter block.
func (s *skillRegistrySource) LoadInstructions(ctx context.Context, name string) (string, error) {
	archive, err := s.load(ctx, name)
	if err != nil {
		return "", err
	}
	file, ok := archive.files["SKILL.md"]
	if !ok {
		return "", fmt.Errorf("%w: skill %q has no SKILL.md at the root of its filesystem", ErrSkillNotFound, name)
	}
	content, err := readZipFile(file, maxResourceSize)
	if err != nil {
		return "", fmt.Errorf("read SKILL.md of skill %q: %w", name, err)
	}
	instructions, err := instructionsOf(content)
	if err != nil {
		return "", fmt.Errorf("skill %q: %w", name, err)
	}
	return instructions, nil
}

// LoadResource returns a reader over one file of the skill's filesystem.
// resourcePath is relative to the skill root and must live under "assets/",
// "references/" or "scripts/".
func (s *skillRegistrySource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	cleanPath, err := validateResourcePath(resourcePath)
	if err != nil {
		return nil, err
	}
	archive, err := s.load(ctx, name)
	if err != nil {
		return nil, err
	}
	file, ok := archive.files[cleanPath]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrResourceNotFound, cleanPath)
	}
	reader, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("open resource %q of skill %q: %w", cleanPath, name, err)
	}
	return reader, nil
}

// ListResources returns the paths, relative to the skill root and in
// lexicographic order, of the files under subpath. An empty subpath or "."
// lists every file under "assets/", "references/" and "scripts/"; any other
// subpath must itself sit under one of those directories.
func (s *skillRegistrySource) ListResources(ctx context.Context, name, subpath string) ([]string, error) {
	cleanPath := path.Clean(subpath)
	isRoot := subpath == "" || cleanPath == "."
	if !isRoot && !slices.Contains(skillResourceDirs, strings.SplitN(cleanPath, "/", 2)[0]) {
		return nil, fmt.Errorf("%w: %q must be empty, root (.), or within 'assets/', 'references/', or 'scripts/'", ErrInvalidResourcePath, subpath)
	}

	archive, err := s.load(ctx, name)
	if err != nil {
		return nil, err
	}
	if isRoot {
		// Listing the root of a skill that simply has no resources is not an
		// error, mirroring the filesystem source.
		var resources []string
		for _, dir := range skillResourceDirs {
			resources = append(resources, archive.under(dir)...)
		}
		slices.Sort(resources)
		return resources, nil
	}

	resources := archive.under(cleanPath)
	if len(resources) == 0 {
		return nil, fmt.Errorf("%w: %q", ErrResourceNotFound, cleanPath)
	}
	return resources, nil
}

// load fetches a skill and opens its zipped filesystem.
func (s *skillRegistrySource) load(ctx context.Context, name string) (*skillArchive, error) {
	sk, err := s.client.GetSkill(ctx, name)
	if err != nil {
		return nil, fmt.Errorf("get skill %q: %w", name, err)
	}
	archive, err := newSkillArchive(sk.ZippedFilesystem)
	if err != nil {
		return nil, fmt.Errorf("skill %q: %w", name, err)
	}
	return archive, nil
}

// skillArchive is a read-only view over a skill's zipped filesystem.
type skillArchive struct {
	// files maps a cleaned, slash-separated path relative to the skill root to
	// the archive entry holding it. Directory entries are not included.
	files map[string]*zip.File
	// paths holds the keys of files in lexicographic order.
	paths []string
}

// newSkillArchive indexes the files of a skill's zipped filesystem. Entries
// whose names escape the skill root are dropped rather than reported: the
// archive is untrusted input, and a malformed entry must not make an otherwise
// usable skill unreadable.
func newSkillArchive(zipped []byte) (*skillArchive, error) {
	if len(zipped) == 0 {
		return nil, fmt.Errorf("%w: empty zipped filesystem", ErrSkillNotFound)
	}
	reader, err := zip.NewReader(bytes.NewReader(zipped), int64(len(zipped)))
	if err != nil {
		return nil, fmt.Errorf("read zipped filesystem: %w", err)
	}

	archive := &skillArchive{files: make(map[string]*zip.File, len(reader.File))}
	for _, file := range reader.File {
		if file.FileInfo().IsDir() {
			continue
		}
		name := path.Clean(file.Name) // Also drops any "./" prefix.
		if name == "." || name == ".." || path.IsAbs(name) || strings.HasPrefix(name, "../") {
			continue
		}
		archive.files[name] = file
		archive.paths = append(archive.paths, name)
	}
	slices.Sort(archive.paths)
	return archive, nil
}

// under returns the paths of the files at or below dir, in lexicographic order.
func (a *skillArchive) under(dir string) []string {
	var found []string
	if _, ok := a.files[dir]; ok {
		found = append(found, dir)
	}
	prefix := dir + "/" // dir is cleaned, so it never ends in a slash.
	start, _ := slices.BinarySearch(a.paths, prefix)
	for _, p := range a.paths[start:] {
		if !strings.HasPrefix(p, prefix) {
			break // The paths are sorted, so nothing further can match.
		}
		found = append(found, p)
	}
	return found
}

// frontmatterOf projects a registered skill onto the frontmatter of the
// SKILL.md it stands for. The skill's ID is used as the name, so that it
// matches the folder the skill is mounted under.
func frontmatterOf(sk *Skill) *Frontmatter {
	return &Frontmatter{
		Name:          sk.ID(),
		Description:   sk.Description,
		License:       sk.License,
		Compatibility: sk.Compatibility,
	}
}

// isUsableSkill reports whether a skill in the given state can be read. An
// empty state is assumed usable, as it only means the API did not report one.
func isUsableSkill(state string) bool {
	return state == "" || state == SkillStateActive
}

// instructionsOf returns the Markdown body of a SKILL.md file, dropping the
// leading YAML frontmatter block.
//
// Unlike [ParseBytes] it does not decode or validate that block, so a skill
// whose frontmatter carries fields this package does not model — or a display
// name that is not a valid skill name — still yields its instructions.
func instructionsOf(content []byte) (string, error) {
	rest, ok := trimSeparator(content)
	if !ok {
		return "", fmt.Errorf("%w: SKILL.md must open with a '---' separator line", ErrInvalidFrontmatter)
	}
	for {
		// rest always starts at a line boundary, so a separator prefix here is
		// a whole separator line and thus closes the frontmatter block.
		if body, ok := trimSeparator(rest); ok {
			return string(body), nil
		}
		_, remainder, found := bytes.Cut(rest, []byte("\n"))
		if !found {
			return "", fmt.Errorf("%w: SKILL.md has no closing '---' separator line", ErrInvalidFrontmatter)
		}
		rest = remainder
	}
}

// trimSeparator strips a leading frontmatter separator line, in either Unix or
// Windows form, reporting whether one was present.
func trimSeparator(content []byte) ([]byte, bool) {
	for _, separator := range [][]byte{frontmatterSeparator, frontmatterSeparatorWin} {
		if rest, ok := bytes.CutPrefix(content, separator); ok {
			return rest, true
		}
	}
	return nil, false
}

// validateResourcePath cleans resourcePath and checks that it stays within the
// directories that may hold skill resources.
func validateResourcePath(resourcePath string) (string, error) {
	cleanPath := path.Clean(resourcePath)
	for _, dir := range skillResourceDirs {
		if strings.HasPrefix(cleanPath, dir+"/") {
			return cleanPath, nil
		}
	}
	return "", fmt.Errorf("%w: %q must be within 'assets/', 'references/', or 'scripts/' (relative to the skill root)", ErrInvalidResourcePath, resourcePath)
}

// readZipFile reads one archive entry in full, refusing to decompress more than
// limit bytes so that a malformed or hostile archive cannot exhaust memory.
func readZipFile(file *zip.File, limit int64) ([]byte, error) {
	reader, err := file.Open()
	if err != nil {
		return nil, err
	}
	defer func() { _ = reader.Close() }()

	content, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(content)) > limit {
		return nil, fmt.Errorf("decompressed size exceeds the %d byte limit", limit)
	}
	return content, nil
}
