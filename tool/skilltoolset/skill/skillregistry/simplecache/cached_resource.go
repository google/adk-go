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

package simplecache

import (
	"fmt"
	"io"
	"strings"
	"sync"

	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/tool/skilltoolset/skill"
	"google.golang.org/adk/v2/tool/skilltoolset/skill/skillregistry"
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

type resources struct {
	client skillregistry.Client
	mu     sync.RWMutex
	// loaded keeps information whether the resources have been loaded - it may happen that a skill has no resources.
	loaded    bool
	res       []*resource
	pathToRes map[string]*resource
}

type resource struct {
	path    string
	content []byte
}

func newResources(client skillregistry.Client) *resources {
	return &resources{
		client: client,
		mu:     sync.RWMutex{},
	}
}

// loadResources cleans internal data and re-creates it using the provided rev
func (r *resources) loadResources(rev string) error {
	// clear first
	r.loaded = false
	r.res = make([]*resource, 0)
	r.pathToRes = make(map[string]*resource)

	zip, err := r.client.GetZip(rev)
	if err != nil {
		return fmt.Errorf("cannot GetZip: %w", err)
	}
	for _, f := range zip.File {
		// Skip directory entries; only files are served as resources.
		if strings.HasSuffix(f.Name, "/") {
			continue
		}
		// The archive is untrusted input. Drop any entry whose name would
		// escape the skill root (a zip-slip payload) and index the survivors
		// under their cleaned path, so a later lookup of the same cleaned path
		// agrees with what was stored.
		cleanPath, err := safeResourcePath(f.Name)
		if err != nil {
			continue
		}
		fr, err := f.Open()
		if err != nil {
			return fmt.Errorf("cannot open an archive entry: %w", err)
		}
		content, err := io.ReadAll(fr)
		closeErr := fr.Close()
		if err != nil {
			return fmt.Errorf("cannot read an archive entry: %w", err)
		}
		if closeErr != nil {
			return fmt.Errorf("cannot close an archive entry: %w", closeErr)
		}
		newRes := &resource{
			path:    cleanPath,
			content: content,
		}
		r.res = append(r.res, newRes)
		r.pathToRes[cleanPath] = newRes
	}
	r.loaded = true
	return nil
}

// listResources lists the files of a skill's filesystem under subpath. An empty
// subpath, ".", "/" or "\" names the root and lists every file. Otherwise
// subpath names a directory (or a single file): a file is returned if it equals
// subpath or sits anywhere below it.
//
// Matching is path-segment aware, not a raw string prefix, so "script" does not
// match "scripts/run.sh". subpath is validated through [safeResourcePath]; a
// traversal or absolute subpath returns [skill.ErrInvalidResourcePath].
func (r *resources) listResources(rev, subpath string) ([]string, error) {
	// An empty, "." , "/" or "\" subpath means the whole filesystem.
	isRoot := subpath == "" || subpath == "." || subpath == "/" || subpath == "\\"
	prefix := ""
	if !isRoot {
		clean, err := safeResourcePath(subpath)
		if err != nil {
			return nil, err
		}
		prefix = clean
	}

	r.mu.RLock()

	if !r.loaded {
		r.mu.RUnlock()
		r.mu.Lock()
		// still not loaded?
		if !r.loaded {
			err := r.loadResources(rev)
			if err != nil {
				r.mu.Unlock()
				return nil, err
			}
		}
		r.mu.Unlock()
		r.mu.RLock()
	}

	defer r.mu.RUnlock()

	res := make([]string, 0)
	for _, entry := range r.res {
		// Paths are cleaned and slash-separated at load time, so a plain
		// segment-aware comparison is enough here.
		if isRoot || entry.path == prefix || strings.HasPrefix(entry.path, prefix+"/") {
			res = append(res, entry.path)
		}
	}

	return res, nil
}

func (r *resources) getResource(rev, path string) (*resource, error) {
	cleanPath, err := safeResourcePath(path)
	if err != nil {
		return nil, err
	}

	r.mu.RLock()
	if r.loaded {
		defer r.mu.RUnlock()
	} else {
		r.mu.RUnlock()
		// even if the whole cache is swapped now for the new version continue with the values we have.
		r.mu.Lock()
		defer r.mu.Unlock()
		// still not loaded?
		if !r.loaded {
			err := r.loadResources(rev)
			if err != nil {
				return nil, err
			}
		}
	}
	// loaded
	res, ok := r.pathToRes[cleanPath]
	if !ok {
		return nil, fmt.Errorf("%w: %q", skill.ErrResourceNotFound, cleanPath)
	}

	return res, nil
}
