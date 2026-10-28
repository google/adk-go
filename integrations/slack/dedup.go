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

package slack

import (
	"context"
	"sync"
	"time"

	"github.com/hashicorp/golang-lru/v2/expirable"
)

// Deduper drops events Slack redelivers, such as retries after a slow ack.
type Deduper interface {
	// Seen records key and reports whether it was already recorded within the window.
	Seen(ctx context.Context, key string) (bool, error)
}

type lruDeduper struct {
	// mu makes check-and-record one step. The LRU locks each call on its own
	// and has no contains-or-add, so two concurrent deliveries could both
	// see the key as new.
	mu  sync.Mutex
	lru *expirable.LRU[string, struct{}]
}

// newLRUDeduper returns an in-memory Deduper remembering up to size keys for ttl.
func newLRUDeduper(size int, ttl time.Duration) Deduper {
	return &lruDeduper{lru: expirable.NewLRU[string, struct{}](size, nil, ttl)}
}

func (d *lruDeduper) Seen(_ context.Context, key string) (bool, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	// Peek, not Contains: Contains reports an expired key as present until the
	// background sweep deletes it.
	if _, ok := d.lru.Peek(key); ok {
		return true, nil
	}
	d.lru.Add(key, struct{}{})
	return false, nil
}
