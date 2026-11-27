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

// Package gcsskill provides a Google Cloud Storage [skill.Source].
//
// Skills live as immediate subdirectories under an optional base prefix within
// a bucket, matching the directory layout expected by
// [skill.NewFileSystemSource]:
//
//	<basePath>/skill-1/
//	    SKILL.md
//	    assets/
//	<basePath>/skill-2/
//	    SKILL.md
//	    references/
//	    scripts/
//
// A source reads the bucket on every call. A skill toolset lists the
// frontmatters on every model request, so a toolset that is given the source
// as it is reads the bucket on every model request, and a request to Cloud
// Storage that fails then fails the model request. When serving live traffic,
// wrap the source in [skill.WithFrontmatterPreloadSource] or
// [skill.WithCompletePreloadSource]. The latter reads every skill with its
// resources once, as adk-python does when it loads skills from a bucket; one
// skill that cannot be read, for example because a resource is larger than
// its 10 MiB limit, then fails the load of them all.
//
//	client, err := storage.NewClient(ctx)
//	if err != nil {
//		return err
//	}
//	defer client.Close()
//
//	source, err := gcsskill.NewSource(gcsskill.Config{
//		Client:   client,
//		Bucket:   "my-skills-bucket",
//		BasePath: "skills",
//	})
//	if err != nil {
//		return err
//	}
//	preloaded, reload, err := skill.WithCompletePreloadSource(ctx, source)
package gcsskill

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"path"
	"slices"
	"strings"
	"sync"
	"unicode/utf8"

	"cloud.google.com/go/storage"
	"google.golang.org/api/iterator"

	"google.golang.org/adk/v2/tool/skilltoolset/skill"
)

// maxSkillFileSize is the largest SKILL.md object whose instructions are
// loaded, matching the 10 MiB resource cap enforced by package skill.
const maxSkillFileSize = 10 * 1024 * 1024

var resourceDirs = []string{"references", "assets", "scripts"}

// Config configures a Google Cloud Storage [skill.Source].
type Config struct {
	// Client reads the bucket. Required. The caller creates it, which is where
	// credentials, the quota project and the endpoint are chosen, and closes
	// it once the source is no longer used.
	Client *storage.Client

	// Bucket is the name of the Cloud Storage bucket holding skills. Required.
	Bucket string

	// BasePath is an optional object-name prefix within Bucket under which
	// skill directories live, such as "skills" or "team-a/skills". Slashes at
	// its start and end are ignored. When empty, skills are read from the
	// bucket root.
	BasePath string

	// OnListInvalidSkill, when non-nil, is called by ListFrontmatters for each
	// skill directory under BasePath that is invalid: its SKILL.md does not
	// parse or validate, or its name is not the directory's. It receives the
	// name of the directory and the error. Returning nil skips that skill and
	// continues listing; returning a non-nil error aborts ListFrontmatters
	// with that error. When nil, the first invalid skill fails
	// ListFrontmatters.
	//
	// It is called on every listing, and may be called from several
	// goroutines when ListFrontmatters is.
	//
	// A directory without a SKILL.md object is not a skill and is ignored
	// without a call. A request to Cloud Storage that fails, for example
	// because access is denied or the network is down, says nothing about the
	// skill, so it always aborts ListFrontmatters and is not offered here.
	OnListInvalidSkill func(name string, err error) error
}

type gcsSource struct {
	bucket             *storage.BucketHandle
	basePrefix         string
	onListInvalidSkill func(name string, err error) error

	mu sync.Mutex
	// validSkills holds the names of the skills whose SKILL.md was found valid
	// when it was last read. LoadResource and ListResources serve only skills,
	// so that the resource directories of a directory that is not a skill
	// cannot be read. For a skill in this set they skip the request that
	// checks its SKILL.md, which otherwise comes before every resource.
	validSkills map[string]bool
}

var _ skill.Source = (*gcsSource)(nil)

// NewSource creates a [skill.Source] backed by the Cloud Storage bucket
// configured in cfg. It issues no request.
//
// A skill is named by its directory. Per-skill methods take the name of one
// directory and return [skill.ErrInvalidSkillName] for anything else before
// issuing a request (note that [skill.NewMergedSource] falls through only on
// [skill.ErrSkillNotFound], so such a name ends a merged lookup). Of a
// SKILL.md object only the frontmatter is read, unless the instructions are
// asked for; LoadInstructions fails if the object is larger than 10 MiB.
//
// Resources are served only for a skill whose SKILL.md is valid. The source
// remembers the skills that it found valid, so that it does not read a
// SKILL.md again before every resource. A skill that is deleted or made
// invalid may therefore serve its resources until its SKILL.md is read again,
// which ListFrontmatters, LoadFrontmatter and LoadInstructions do.
//
// A resource is named by the path of its object below the skill directory, as
// adk-python names it: a backslash separates like a slash, and empty and "."
// segments are dropped, so that the object "references/sub//doc.md" is the
// resource "references/sub/doc.md". adk-python refuses an object whose name
// below 'references/', 'assets/' or 'scripts/' has a ".." segment, or starts
// with a separator or with a drive as Windows writes one ("c:"), and two
// objects that name the same resource. Such objects fail the methods that
// list the directory with [skill.ErrInvalidResourcePath], so that a bucket
// gives the same skills to both.
//
// The returned source is safe for concurrent use by multiple goroutines.
func NewSource(cfg Config) (skill.Source, error) {
	if cfg.Client == nil {
		return nil, errors.New("storage client must be provided")
	}
	if cfg.Bucket == "" {
		return nil, errors.New("bucket name must be provided")
	}
	basePrefix := strings.Trim(cfg.BasePath, "/")
	if basePrefix != "" {
		basePrefix += "/"
	}
	return &gcsSource{
		bucket:             cfg.Client.Bucket(cfg.Bucket),
		basePrefix:         basePrefix,
		onListInvalidSkill: cfg.OnListInvalidSkill,
		validSkills:        make(map[string]bool),
	}, nil
}

// ListFrontmatters lists the immediate directories under BasePath and parses
// the SKILL.md frontmatter of each. It does not traverse recursively.
// Directories without a SKILL.md object are ignored; an invalid skill fails
// the listing unless Config.OnListInvalidSkill returns nil.
func (s *gcsSource) ListFrontmatters(ctx context.Context) ([]*skill.Frontmatter, error) {
	q := &storage.Query{
		Prefix:    s.basePrefix,
		Delimiter: "/",
	}
	if err := q.SetAttrSelection([]string{"Name"}); err != nil {
		return nil, fmt.Errorf("select object attributes: %w", err)
	}

	iter := s.bucket.Objects(ctx, q)
	var prefixes []string
	for {
		attrs, err := iter.Next()
		if errors.Is(err, iterator.Done) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("list skill prefixes: %w", err)
		}
		if attrs.Prefix == "" {
			continue // Object directly under basePrefix; skills must be directories.
		}
		prefixes = append(prefixes, attrs.Prefix)
	}
	// Read each directory once, in case one is reported on several pages of
	// the listing.
	slices.Sort(prefixes)
	prefixes = slices.Compact(prefixes)

	var frontmatters []*skill.Frontmatter
	for _, prefix := range prefixes {
		name := strings.TrimSuffix(strings.TrimPrefix(prefix, s.basePrefix), "/")
		frontmatter, _, err := s.readSkill(ctx, name, false)
		if errors.Is(err, skill.ErrSkillNotFound) {
			continue // Directory doesn't contain SKILL.md - not a skill.
		}
		if err != nil {
			// Fail closed. Only a skill that was read and found invalid is
			// offered to OnListInvalidSkill for skipping; any other error is
			// a failed request, which leaves the skill's state unknown.
			invalid := errors.Is(err, skill.ErrInvalidFrontmatter) || errors.Is(err, skill.ErrInvalidSkillName)
			if s.onListInvalidSkill == nil || !invalid {
				return nil, err
			}
			if err := s.onListInvalidSkill(name, err); err != nil {
				return nil, err
			}
			continue
		}
		frontmatters = append(frontmatters, frontmatter)
	}

	// Forget the skills that are no longer in the bucket: readSkill has
	// brought the set up to date for every directory that is.
	valid := make(map[string]bool, len(frontmatters))
	for _, frontmatter := range frontmatters {
		valid[frontmatter.Name] = true
	}
	s.mu.Lock()
	s.validSkills = valid
	s.mu.Unlock()
	return frontmatters, nil
}

// LoadFrontmatter reads and validates the SKILL.md frontmatter for name.
func (s *gcsSource) LoadFrontmatter(ctx context.Context, name string) (*skill.Frontmatter, error) {
	frontmatter, _, err := s.loadSkill(ctx, name, false)
	return frontmatter, err
}

// LoadInstructions reads SKILL.md for name and returns the markdown body
// following the closing frontmatter delimiter, without the white space around
// it.
func (s *gcsSource) LoadInstructions(ctx context.Context, name string) (string, error) {
	_, instructions, err := s.loadSkill(ctx, name, true)
	return instructions, err
}

// LoadResource opens a stream for resourcePath within skill name. Access is
// restricted to resources under 'references/', 'assets/', or 'scripts/'.
//
// When two objects name the resource, the one whose name is exactly the path
// of the resource is served; ListResources reports the pair as an error.
func (s *gcsSource) LoadResource(ctx context.Context, name, resourcePath string) (io.ReadCloser, error) {
	if err := s.requireSkill(ctx, name); err != nil {
		return nil, err
	}

	cleanPath := path.Clean(resourcePath)
	topDir, rest, isFile := strings.Cut(cleanPath, "/")
	if !isFile || !slices.Contains(resourceDirs, topDir) {
		return nil, fmt.Errorf("%w: %q must be within 'references/', 'assets/', or 'scripts/' (relative to skill directory)", skill.ErrInvalidResourcePath, resourcePath)
	}
	// No resource has a backslash in its path or a drive at its start, while
	// an object may: without this check such an object would be served,
	// though it is not a resource or is one under another name.
	if strings.Contains(cleanPath, `\`) || startsWithDrive(rest) {
		return nil, fmt.Errorf("%w: %q", skill.ErrResourceNotFound, cleanPath)
	}

	// Most resources are objects named exactly by their path, so that name is
	// tried first. Only when there is no such object is the directory listed
	// for one that names the resource in another form, which costs a request.
	objectName := s.basePrefix + name + "/" + cleanPath
	reader, err := s.bucket.Object(objectName).NewReader(ctx)
	if errors.Is(err, storage.ErrObjectNotExist) {
		objects, _, listErr := s.listResourceDir(ctx, name, topDir)
		if listErr != nil {
			return nil, listErr
		}
		if otherName, ok := objects[cleanPath]; ok {
			objectName = otherName
			reader, err = s.bucket.Object(objectName).NewReader(ctx)
		}
	}
	// The object may also be deleted after it was listed.
	if errors.Is(err, storage.ErrObjectNotExist) {
		return nil, fmt.Errorf("%w: %q", skill.ErrResourceNotFound, cleanPath)
	}
	if err != nil {
		return nil, fmt.Errorf("open resource %q: %w", objectName, err)
	}
	return reader, nil
}

// ListResources returns resource paths relative to skill name under subpath,
// sorted. When subpath is empty or ".", it lists 'references/', 'assets/', and
// 'scripts/', in this order.
func (s *gcsSource) ListResources(ctx context.Context, name, subpath string) ([]string, error) {
	if err := s.requireSkill(ctx, name); err != nil {
		return nil, err
	}

	cleanPath := path.Clean(subpath)
	if cleanPath == "." {
		var resources []string
		for _, dir := range resourceDirs {
			objects, _, err := s.listResourceDir(ctx, name, dir)
			if err != nil {
				return nil, err
			}
			resources = append(resources, slices.Sorted(maps.Keys(objects))...)
		}
		return resources, nil
	}

	topDir, _, _ := strings.Cut(cleanPath, "/")
	if !slices.Contains(resourceDirs, topDir) {
		return nil, fmt.Errorf("%w: %q must be empty, root (.), or within 'references/', 'assets/', or 'scripts/'", skill.ErrInvalidResourcePath, subpath)
	}

	objects, dirs, err := s.listResourceDir(ctx, name, topDir)
	if err != nil {
		return nil, err
	}
	// The subpath exists if it is a file, a directory marker, or an ancestor
	// of either: Cloud Storage has no directories of its own.
	subdirPrefix := cleanPath + "/"
	found := dirs[cleanPath]
	for dir := range dirs {
		if strings.HasPrefix(dir, subdirPrefix) {
			found = true
		}
	}
	var matched []string
	for _, file := range slices.Sorted(maps.Keys(objects)) {
		if file == cleanPath || strings.HasPrefix(file, subdirPrefix) {
			matched = append(matched, file)
			found = true
		}
	}
	if !found {
		return nil, fmt.Errorf("%w: %q", skill.ErrResourceNotFound, cleanPath)
	}
	return matched, nil
}

// loadSkill reads the skill that a caller names. The name must be that of one
// directory, which is checked before any request is issued.
func (s *gcsSource) loadSkill(ctx context.Context, name string, withInstructions bool) (*skill.Frontmatter, string, error) {
	if name == "" || name == "." || name == ".." || strings.ContainsAny(name, "/\\\x00") {
		return nil, "", fmt.Errorf("%w: %q is not the name of one directory", skill.ErrInvalidSkillName, name)
	}
	return s.readSkill(ctx, name, withInstructions)
}

// requireSkill returns an error unless name is a valid skill. It issues a
// request only if the skill is not remembered as valid.
func (s *gcsSource) requireSkill(ctx context.Context, name string) error {
	s.mu.Lock()
	valid := s.validSkills[name]
	s.mu.Unlock()
	if valid {
		return nil
	}
	_, _, err := s.loadSkill(ctx, name, false)
	return err
}

// readSkill reads the SKILL.md of the directory name and returns its
// validated frontmatter. The rest of the object, the instructions, is read
// and returned only withInstructions. It remembers whether the skill is valid.
func (s *gcsSource) readSkill(ctx context.Context, name string, withInstructions bool) (_ *skill.Frontmatter, _ string, err error) {
	defer func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		if err != nil {
			// Also after a request that failed: the skill is then checked
			// again before its next resource is served.
			delete(s.validSkills, name)
			return
		}
		s.validSkills[name] = true
	}()

	objectName := s.basePrefix + name + "/SKILL.md"
	object, err := s.bucket.Object(objectName).NewReader(ctx)
	if errors.Is(err, storage.ErrObjectNotExist) {
		return nil, "", fmt.Errorf("%w: %q", skill.ErrSkillNotFound, name)
	}
	if err != nil {
		return nil, "", fmt.Errorf("open %q: %w", objectName, err)
	}
	defer func() {
		_ = object.Close() // Ignore error as read success is what matters.
	}()

	// One byte more than is allowed may be read, to tell an object of exactly
	// the maximum size from a larger one.
	limited := &io.LimitedReader{R: object, N: maxSkillFileSize + 1}
	failure := &readFailure{reader: limited}
	reader := bufio.NewReader(failure)
	frontmatter, err := skill.Parse(reader)
	if err != nil {
		// skill.Parse reports a read that failed the same way as a file that
		// is malformed. Only the latter makes the skill invalid.
		if failure.err != nil {
			return nil, "", fmt.Errorf("read %q: %w", objectName, failure.err)
		}
		return nil, "", fmt.Errorf("%w: %q: %w", skill.ErrInvalidFrontmatter, objectName, err)
	}
	if frontmatter.Name != name {
		return nil, "", fmt.Errorf("%w: name in SKILL.md (%q) does not match directory name (%q)", skill.ErrInvalidSkillName, frontmatter.Name, name)
	}
	if !withInstructions {
		return frontmatter, "", nil
	}

	instructions, err := io.ReadAll(reader)
	if err != nil {
		return nil, "", fmt.Errorf("read instructions of %q: %w", objectName, err)
	}
	if limited.N == 0 {
		return nil, "", fmt.Errorf("%q exceeds %d bytes limit", objectName, maxSkillFileSize)
	}
	return frontmatter, strings.TrimSpace(string(instructions)), nil
}

// readFailure is a reader that remembers the error with which a read of the
// reader it wraps failed. The end of the data is not a failure.
type readFailure struct {
	reader io.Reader
	err    error
}

func (f *readFailure) Read(p []byte) (int, error) {
	n, err := f.reader.Read(p)
	if err != nil && !errors.Is(err, io.EOF) {
		f.err = err
	}
	return n, err
}

// listResourceDir lists the objects of one resource directory of skill name.
// It returns the name of the object of each resource, keyed by the path of the
// resource relative to the skill, and the set of directories that
// directory-marker objects (names ending in a slash) declare.
func (s *gcsSource) listResourceDir(ctx context.Context, name, dir string) (map[string]string, map[string]bool, error) {
	dirPrefix := s.basePrefix + name + "/" + dir + "/"
	q := &storage.Query{Prefix: dirPrefix}
	if err := q.SetAttrSelection([]string{"Name"}); err != nil {
		return nil, nil, fmt.Errorf("select object attributes: %w", err)
	}

	iter := s.bucket.Objects(ctx, q)
	objects := make(map[string]string)
	dirs := make(map[string]bool)
	for {
		attrs, err := iter.Next()
		if errors.Is(err, iterator.Done) {
			break
		}
		if err != nil {
			return nil, nil, fmt.Errorf("list resource objects under %q: %w", dirPrefix, err)
		}
		rel := strings.TrimPrefix(attrs.Name, dirPrefix)
		if rel == "" {
			dirs[dir] = true // The marker object of the directory itself.
			continue
		}
		resourcePath, ok := cleanObjectPath(rel)
		if strings.HasSuffix(rel, "/") {
			// A directory marker holds no resource. One whose name is not
			// accepted declares nothing and is ignored.
			if ok {
				dirs[dir+"/"+resourcePath] = true
			}
			continue
		}
		if !ok {
			return nil, nil, fmt.Errorf("%w: object %q: its name below %q must not have a '..' segment or start with a separator or a drive", skill.ErrInvalidResourcePath, attrs.Name, dirPrefix)
		}
		resourcePath = dir + "/" + resourcePath
		if otherName, ok := objects[resourcePath]; ok {
			return nil, nil, fmt.Errorf("%w: objects %q and %q both name the resource %q", skill.ErrInvalidResourcePath, otherName, attrs.Name, resourcePath)
		}
		objects[resourcePath] = attrs.Name
	}
	return objects, dirs, nil
}

// cleanObjectPath returns the path of the resource that an object names,
// given the name of the object relative to its resource directory. As in
// adk-python, a backslash separates like a slash, and empty and "." segments
// are dropped. It reports false for a name that adk-python refuses: one with
// a ".." segment, one that starts with a separator or a drive, and one that
// has no other segment than ".".
func cleanObjectPath(rel string) (string, bool) {
	rel = strings.ReplaceAll(rel, `\`, "/")
	if strings.HasPrefix(rel, "/") || slices.Contains(strings.Split(rel, "/"), "..") {
		return "", false
	}
	clean := path.Clean(rel)
	return clean, clean != "." && !startsWithDrive(clean)
}

// startsWithDrive reports whether p starts with a drive as Windows writes
// one: any one character and a colon, as in "c:notes.md". adk-python refuses
// such a path, because it may later be joined to a directory on a disk.
func startsWithDrive(p string) bool {
	_, size := utf8.DecodeRuneInString(p)
	return size > 0 && strings.HasPrefix(p[size:], ":")
}
