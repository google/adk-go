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
	"log"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
	agentregistry "google.golang.org/api/agentregistry/v1alpha"
	"google.golang.org/api/option"
)

// skillResourceDirs are the top-level directories of a skill's filesystem that
// hold addressable resources. See
// https://agentskills.io/specification#directory-structure.
var skillResourceDirs = []string{"assets", "references", "scripts"}

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

// NewSkillRegistrySource creates a [skill.Source] backed by the Skill Registry of the
// Gemini Enterprise Agent Platform. Client options are forwarded to
// [NewClient].
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
	log.Printf("ListFrontmatters")
	return s.FindFrontmatters(ctx, "private-kdroste-dice-thrower-04")
}

func (s *SkillRegistrySource) ListFrontmatters2(ctx context.Context) ([]*skill.Frontmatter, error) {
	frs, err := s.client.ListFrontmatters()
	if err != nil {
		return nil, fmt.Errorf("cannot list frontmatters: %w", err)
	}

	res := make([]*skill.Frontmatter, 0)
	for _, f := range frs {
		sf := s.toFrontmatter(f)
		if sf == nil {
			continue
		}
		res = append(res, sf)
	}
	return res, nil
}

func (s *SkillRegistrySource) FindFrontmatters(ctx context.Context, query string) ([]*skill.Frontmatter, error) {
	log.Printf("FindFrontmatters: %v", query)
	frs, err := s.client.FindFrontmatters(query)
	if err != nil {
		return nil, fmt.Errorf("cannot list frontmatters: %w", err)
	}

	res := make([]*skill.Frontmatter, 0)
	for _, f := range frs {
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
func (s *SkillRegistrySource) ListResources(ctx context.Context, name string, subpath string) ([]string, error) {
	log.Printf("ListResources: %v, %v", name, subpath)
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
	log.Printf("LoadFrontmatter: %v", name)
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
	log.Printf("LoadInstructions: %v", name)
	sk, err := s.client.GetSkill(name)
	if err != nil {
		return "", fmt.Errorf("cannot GetSkill: %w", err)
	}
	if sk == nil {
		return "", fmt.Errorf("got nil from GetSkill")
	}
	// if skill.Frontmatter == nil {
	// 	return "", fmt.Errorf("got nil from Frontmatter")
	// }
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
			defer b.Close()
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
func (s *SkillRegistrySource) LoadResource(ctx context.Context, name string, resourcePath string) (io.ReadCloser, error) {
	log.Printf("LoadResource: %v, %v", name, resourcePath)
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

// isUsableSkill reports whether a skill in the given state can be read. An
// empty state is assumed usable, as it only means the API did not report one.
func isUsableSkill(state string) bool {
	return state == "" //agentregistry.Active
}

// // LoadInstructions returns the Markdown body of the skill's SKILL.md, that is
// // everything after the YAML frontmatter block.
// func (s *skillRegistrySource) LoadInstructions(ctx context.Context, name string) (string, error) {
// 	archive, err := s.load(ctx, name)
// 	if err != nil {
// 		return "", err
// 	}
// 	file, ok := archive.files["SKILL.md"]
// 	if !ok {
// 		return "", fmt.Errorf("%w: skill %q has no SKILL.md at the root of its filesystem", skill.ErrSkillNotFound, name)
// 	}
// 	content, err := readZipFile(file, maxResourceSize)
// 	if err != nil {
// 		return "", fmt.Errorf("read SKILL.md of skill %q: %w", name, err)
// 	}
// 	instructions, err := instructionsOf(content)
// 	if err != nil {
// 		return "", fmt.Errorf("skill %q: %w", name, err)
// 	}
// 	return instructions, nil
// }

// // LoadResource returns a reader over one file of the skill's filesystem.
// // resourcePath is relative to the skill root and must live under "assets/",
// // "references/" or "scripts/".
// func (s *skillRegistrySource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
// 	cleanPath, err := validateResourcePath(resourcePath)
// 	if err != nil {
// 		return nil, err
// 	}
// 	archive, err := s.load(ctx, name)
// 	if err != nil {
// 		return nil, err
// 	}
// 	file, ok := archive.files[cleanPath]
// 	if !ok {
// 		return nil, fmt.Errorf("%w: %q", skill.ErrResourceNotFound, cleanPath)
// 	}
// 	reader, err := file.Open()
// 	if err != nil {
// 		return nil, fmt.Errorf("open resource %q of skill %q: %w", cleanPath, name, err)
// 	}
// 	return reader, nil
// }

// // ListResources returns the paths, relative to the skill root and in
// // lexicographic order, of the files under subpath. An empty subpath or "."
// // lists every file under "assets/", "references/" and "scripts/"; any other
// // subpath must itself sit under one of those directories.
// func (s *skillRegistrySource) ListResources(ctx context.Context, name, subpath string) ([]string, error) {
// 	cleanPath := path.Clean(subpath)
// 	isRoot := subpath == "" || cleanPath == "."
// 	if !isRoot && !slices.Contains(skillResourceDirs, strings.SplitN(cleanPath, "/", 2)[0]) {
// 		return nil, fmt.Errorf("%w: %q must be empty, root (.), or within 'assets/', 'references/', or 'scripts/'", skill.ErrInvalidResourcePath, subpath)
// 	}

// 	archive, err := s.load(ctx, name)
// 	if err != nil {
// 		return nil, err
// 	}
// 	if isRoot {
// 		// Listing the root of a skill that simply has no resources is not an
// 		// error, mirroring the filesystem source.
// 		var resources []string
// 		for _, dir := range skillResourceDirs {
// 			resources = append(resources, archive.under(dir)...)
// 		}
// 		slices.Sort(resources)
// 		return resources, nil
// 	}

// 	resources := archive.under(cleanPath)
// 	if len(resources) == 0 {
// 		return nil, fmt.Errorf("%w: %q", skill.ErrResourceNotFound, cleanPath)
// 	}
// 	return resources, nil
// }

// // load fetches a skill and opens its zipped filesystem.
// func (s *skillRegistrySource) load(ctx context.Context, name string) (*skillArchive, error) {
// 	sk, err := s.client.GetSkill(ctx, name)
// 	if err != nil {
// 		return nil, fmt.Errorf("get skill %q: %w", name, err)
// 	}
// 	archive, err := newSkillArchive(sk.ZippedFilesystem)
// 	if err != nil {
// 		return nil, fmt.Errorf("skill %q: %w", name, err)
// 	}
// 	return archive, nil
// }

// // skillArchive is a read-only view over a skill's zipped filesystem.
// type skillArchive struct {
// 	// files maps a cleaned, slash-separated path relative to the skill root to
// 	// the archive entry holding it. Directory entries are not included.
// 	files map[string]*zip.File
// 	// paths holds the keys of files in lexicographic order.
// 	paths []string
// }

// // newSkillArchive indexes the files of a skill's zipped filesystem. Entries
// // whose names escape the skill root are dropped rather than reported: the
// // archive is untrusted input, and a malformed entry must not make an otherwise
// // usable skill unreadable.
// func newSkillArchive(zipped []byte) (*skillArchive, error) {
// 	if len(zipped) == 0 {
// 		return nil, fmt.Errorf("%w: empty zipped filesystem", skill.ErrSkillNotFound)
// 	}
// 	reader, err := zip.NewReader(bytes.NewReader(zipped), int64(len(zipped)))
// 	if err != nil {
// 		return nil, fmt.Errorf("read zipped filesystem: %w", err)
// 	}

// 	archive := &skillArchive{files: make(map[string]*zip.File, len(reader.File))}
// 	for _, file := range reader.File {
// 		if file.FileInfo().IsDir() {
// 			continue
// 		}
// 		name := path.Clean(file.Name) // Also drops any "./" prefix.
// 		if name == "." || name == ".." || path.IsAbs(name) || strings.HasPrefix(name, "../") {
// 			continue
// 		}
// 		archive.files[name] = file
// 		archive.paths = append(archive.paths, name)
// 	}
// 	slices.Sort(archive.paths)
// 	return archive, nil
// }

// // under returns the paths of the files at or below dir, in lexicographic order.
// func (a *skillArchive) under(dir string) []string {
// 	var found []string
// 	if _, ok := a.files[dir]; ok {
// 		found = append(found, dir)
// 	}
// 	prefix := dir + "/" // dir is cleaned, so it never ends in a slash.
// 	start, _ := slices.BinarySearch(a.paths, prefix)
// 	for _, p := range a.paths[start:] {
// 		if !strings.HasPrefix(p, prefix) {
// 			break // The paths are sorted, so nothing further can match.
// 		}
// 		found = append(found, p)
// 	}
// 	return found
// }

// // frontmatterOf projects a registered skill onto the frontmatter of the
// // SKILL.md it stands for. The skill's ID is used as the name, so that it
// // matches the folder the skill is mounted under.
// func frontmatterOf(sk *agentregistry.Skill) *skill.Frontmatter {
// 	if sk == nil {
// 		return nil
// 	}

// 	// sk.Examples and Tags are ignored
// 	return &skill.Frontmatter{
// 		Name:        sk.ID,
// 		Description: sk.Description,
// 	}
// }

// // instructionsOf returns the Markdown body of a SKILL.md file, dropping the
// // leading YAML frontmatter block.
// //
// // Unlike [ParseBytes] it does not decode or validate that block, so a skill
// // whose frontmatter carries fields this package does not model — or a display
// // name that is not a valid skill name — still yields its instructions.
// func instructionsOf(content []byte) (string, error) {
// 	rest, ok := trimSeparator(content)
// 	if !ok {
// 		return "", fmt.Errorf("%w: SKILL.md must open with a '---' separator line", skill.ErrInvalidFrontmatter)
// 	}
// 	for {
// 		// rest always starts at a line boundary, so a separator prefix here is
// 		// a whole separator line and thus closes the frontmatter block.
// 		if body, ok := trimSeparator(rest); ok {
// 			return string(body), nil
// 		}
// 		_, remainder, found := bytes.Cut(rest, []byte("\n"))
// 		if !found {
// 			return "", fmt.Errorf("%w: SKILL.md has no closing '---' separator line", skill.ErrInvalidFrontmatter)
// 		}
// 		rest = remainder
// 	}
// }

// // trimSeparator strips a leading frontmatter separator line, in either Unix or
// // Windows form, reporting whether one was present.
// func trimSeparator(content []byte) ([]byte, bool) {
// 	for _, separator := range [][]byte{frontmatterSeparator, frontmatterSeparatorWin} {
// 		if rest, ok := bytes.CutPrefix(content, separator); ok {
// 			return rest, true
// 		}
// 	}
// 	return nil, false
// }

// // validateResourcePath cleans resourcePath and checks that it stays within the
// // directories that may hold skill resources.
// func validateResourcePath(resourcePath string) (string, error) {
// 	cleanPath := path.Clean(resourcePath)
// 	for _, dir := range skillResourceDirs {
// 		if strings.HasPrefix(cleanPath, dir+"/") {
// 			return cleanPath, nil
// 		}
// 	}
// 	return "", fmt.Errorf("%w: %q must be within 'assets/', 'references/', or 'scripts/' (relative to the skill root)", skill.ErrInvalidResourcePath, resourcePath)
// }

// // readZipFile reads one archive entry in full, refusing to decompress more than
// // limit bytes so that a malformed or hostile archive cannot exhaust memory.
// func readZipFile(file *zip.File, limit int64) ([]byte, error) {
// 	reader, err := file.Open()
// 	if err != nil {
// 		return nil, err
// 	}
// 	defer func() { _ = reader.Close() }()

// 	content, err := io.ReadAll(io.LimitReader(reader, limit+1))
// 	if err != nil {
// 		return nil, err
// 	}
// 	if int64(len(content)) > limit {
// 		return nil, fmt.Errorf("decompressed size exceeds the %d byte limit", limit)
// 	}
// 	return content, nil
// }

var _ skill.Source = (*SkillRegistrySource)(nil)
