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

package skilltoolset_test

import (
	"fmt"
	"io/fs"
	"strings"
	"sync/atomic"
	"testing"
	"testing/fstest"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool/skilltoolset"
	"google.golang.org/adk/v2/tool/skilltoolset/skill"
)

type countingFS struct {
	fs.FS
	skillOpens atomic.Int64
}

func (c *countingFS) Open(name string) (fs.File, error) {
	if strings.HasSuffix(name, "SKILL.md") {
		c.skillOpens.Add(1)
	}
	return c.FS.Open(name)
}

// BenchmarkSkillToolsetProcessRequest measures skilltoolset.ProcessRequest over
// a filesystem skill source containing 50 skills.
func BenchmarkSkillToolsetProcessRequest(b *testing.B) {
	ctx := b.Context()
	mapFS := make(fstest.MapFS, 50)
	for i := range 50 {
		dir := fmt.Sprintf("skill-%02d", i)
		content := fmt.Sprintf("---\nname: %s\ndescription: Skill number %d for benchmarking.\n---\n# Skill %d\nInstructions here.\n", dir, i, i)
		mapFS[dir+"/SKILL.md"] = &fstest.MapFile{Data: []byte(content)}
	}
	cfs := &countingFS{FS: mapFS}
	src := skill.NewFileSystemSource(cfs)
	ts, err := skilltoolset.New(ctx, skilltoolset.Config{Source: src})
	if err != nil {
		b.Fatalf("skilltoolset.New failed: %v", err)
	}

	cfs.skillOpens.Store(0)
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		req := &model.LLMRequest{}
		if err := ts.ProcessRequest(nil, req); err != nil {
			b.Fatalf("ProcessRequest failed: %v", err)
		}
	}

	b.StopTimer()
	b.ReportMetric(float64(cfs.skillOpens.Load())/float64(b.N), "skill-file-opens/model-call")
}
