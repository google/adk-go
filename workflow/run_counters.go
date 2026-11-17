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

package workflow

import (
	"context"
	"strconv"
	"strings"
	"sync"

	"google.golang.org/adk/v2/agent"
)

// runCounters allocates run ids: the n-th activation of a node name
// under one parent path gets run id strconv.Itoa(n). A workflow run's
// static scheduler and every dynamic sub-scheduler under it share one
// table, so static and dynamic activations of a name count together and
// a retried attempt keeps counting where the previous attempt stopped.
// Mirrors adk-python's run_counters and next_run_id
// (workflow/_dynamic_node_scheduler.py).
//
// The zero value is ready to use. It is safe for concurrent use: the
// consumer allocates static ids while node goroutines allocate ids for
// their dynamic children.
type runCounters struct {
	mu     sync.Mutex
	counts map[string]map[string]int // parent path -> node name -> last run id
}

// next increments the counter of name under parentPath and returns the
// new value: "1" on the first call, then "2", and so on.
func (c *runCounters) next(parentPath, name string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	byName := c.byPath(parentPath)
	byName[name]++
	return strconv.Itoa(byName[name])
}

// seed raises the counter of name under parentPath to at least n, so
// the next call to next returns n+1 or more. Used by Resume to continue
// the count of the activations history shows.
func (c *runCounters) seed(parentPath, name string, n int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	byName := c.byPath(parentPath)
	if byName[name] < n {
		byName[name] = n
	}
}

// reset forgets the counters of parentPath and of every path nested
// under it, so the children of a retried activation start again at
// run id 1. An empty parentPath is a no-op: it would otherwise reset
// the whole table. Mirrors adk-python's reset_for_retry.
func (c *runCounters) reset(parentPath string) {
	if parentPath == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	prefix := parentPath + "/"
	for path := range c.counts {
		if path == parentPath || strings.HasPrefix(path, prefix) {
			delete(c.counts, path)
		}
	}
}

// byPath returns the counters under parentPath, creating them. The
// caller holds c.mu.
func (c *runCounters) byPath(parentPath string) map[string]int {
	if c.counts == nil {
		c.counts = map[string]map[string]int{}
	}
	byName := c.counts[parentPath]
	if byName == nil {
		byName = map[string]int{}
		c.counts[parentPath] = byName
	}
	return byName
}

type runCountersKey struct{}

// withRunCounters returns ctx carrying c, so the dynamic sub-schedulers
// of the node that runs in ctx allocate their children's run ids from
// the scheduler's table.
func withRunCounters(ctx agent.Context, c *runCounters) agent.Context {
	base := context.WithValue(ctx, runCountersKey{}, c)
	return ctx.WithDelta(&agent.CommonContextDelta{
		InvocationContextDelta: &agent.InvocationContextDelta{Context: &base},
	})
}

// runCountersFrom returns the table withRunCounters put on ctx, or nil
// when ctx is nil or carries none.
func runCountersFrom(ctx context.Context) *runCounters {
	if ctx == nil {
		return nil
	}
	c, _ := ctx.Value(runCountersKey{}).(*runCounters)
	return c
}
