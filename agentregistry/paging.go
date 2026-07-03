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

package agentregistry

import (
	"context"
	"iter"
)

// pages returns an iterator that yields every item across all pages, fetching
// subsequent pages on demand via fetch. fetch returns one page of items and the
// next page token (empty when there are no more pages). On error the iterator
// yields a single (nil, err) and stops.
func pages[T any](
	ctx context.Context,
	opts []ListOption,
	fetch func(context.Context, ...ListOption) (items []T, nextPageToken string, err error),
) iter.Seq2[*T, error] {
	return func(yield func(*T, error) bool) {
		token := ""
		for {
			pageOpts := opts
			if token != "" {
				pageOpts = append(append([]ListOption(nil), opts...), WithPageToken(token))
			}

			items, next, err := fetch(ctx, pageOpts...)
			if err != nil {
				yield(nil, err)
				return
			}
			for i := range items {
				if !yield(&items[i], nil) {
					return
				}
			}
			if next == "" {
				return
			}
			token = next
		}
	}
}
