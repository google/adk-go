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
	"log"
	"strings"
	"sync"

	"google.golang.org/adk/v2/tool/skilltoolset/skill/skillregistry"
)

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

	log.Printf("loadResources")
	// clear first
	r.loaded = false
	r.res = make([]*resource, 0)
	r.pathToRes = make(map[string]*resource)

	zip, err := r.client.GetZip(rev)
	if err != nil {
		return fmt.Errorf("cannot GetZip: %w", err)
	}
	for _, f := range zip.File {
		fr, err := f.Open()
		if err != nil {
			return fmt.Errorf("cannot open %v: %w", f.Name, err)
		}
		content, err := io.ReadAll(fr)
		if err != nil {
			return fmt.Errorf("cannot read %v: %w", f.Name, err)
		}
		newRes := &resource{
			path:    f.Name,
			content: content,
		}
		r.res = append(r.res, newRes)
		r.pathToRes[f.Name] = newRes
	}
	r.loaded = true
	log.Printf("loadResources done")
	return nil
}

func (r *resources) listResources(rev, subpath string) ([]string, error) {
	r.mu.RLock()

	if !r.loaded {
		r.mu.RUnlock()
		r.mu.Lock()
		// still not loaded?
		if !r.loaded {
			err := r.loadResources(rev)
			if err != nil {
				return nil, err
			}
		}
		r.mu.Unlock()
		r.mu.RLock()
	}

	defer r.mu.RUnlock()

	res := make([]string, 0)
	for _, r := range r.res {
		if strings.HasPrefix(r.path, subpath) {
			res = append(res, r.path)
		}
	}

	return res, nil
}

func (r *resources) getResource(rev, path string) (*resource, error) {
	log.Printf("getResource %v", path)
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
	res, ok := r.pathToRes[path]
	if !ok {
		return nil, fmt.Errorf("cannot find %v", path)
	}

	log.Printf("getResource done")
	return res, nil
}
