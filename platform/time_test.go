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

package platform_test

import (
	"context"
	"testing"
	"time"

	"google.golang.org/adk/platform"
)

func TestNowDefaultUsesWallClock(t *testing.T) {
	before := time.Now()
	got := platform.Now(context.Background())
	after := time.Now()

	if got.Before(before) || got.After(after) {
		t.Errorf("Now() = %v, want within [%v, %v]", got, before, after)
	}
}

func TestNowNilContext(t *testing.T) {
	// Now must tolerate a nil context and fall back to the wall clock.
	var nilCtx context.Context
	before := time.Now()
	got := platform.Now(nilCtx)
	after := time.Now()

	if got.Before(before) || got.After(after) {
		t.Errorf("Now(nil) = %v, want within [%v, %v]", got, before, after)
	}
}

func TestWithTimeProviderOverridesNow(t *testing.T) {
	fixed := time.Date(2024, time.January, 2, 3, 4, 5, 0, time.UTC)
	ctx := platform.WithTimeProvider(context.Background(), func() time.Time { return fixed })

	if got := platform.Now(ctx); !got.Equal(fixed) {
		t.Errorf("Now() = %v, want %v", got, fixed)
	}
}

func TestWithTimeProviderDerivedContext(t *testing.T) {
	fixed := time.Date(2024, time.January, 2, 3, 4, 5, 0, time.UTC)
	ctx := platform.WithTimeProvider(context.Background(), func() time.Time { return fixed })

	// TODO(kdroste): refactor underlying context
	// A context derived from the provider context must keep the provider.
	derived, cancel := context.WithCancel(ctx)
	defer cancel()

	if got := platform.Now(derived); !got.Equal(fixed) {
		t.Errorf("Now() on derived context = %v, want %v", got, fixed)
	}
}

func TestWithTimeProviderNilFallsBack(t *testing.T) {
	ctx := platform.WithTimeProvider(context.Background(), nil)

	before := time.Now()
	got := platform.Now(ctx)
	after := time.Now()

	if got.Before(before) || got.After(after) {
		t.Errorf("Now() with nil provider = %v, want within [%v, %v]", got, before, after)
	}
}

func TestWithTimeProviderNestedOverride(t *testing.T) {
	outer := time.Date(2024, time.January, 1, 0, 0, 0, 0, time.UTC)
	inner := time.Date(2025, time.June, 7, 8, 9, 10, 0, time.UTC)

	ctx := platform.WithTimeProvider(context.Background(), func() time.Time { return outer })
	ctx = platform.WithTimeProvider(ctx, func() time.Time { return inner })

	if got := platform.Now(ctx); !got.Equal(inner) {
		t.Errorf("Now() = %v, want innermost provider value %v", got, inner)
	}
}

func TestWithTimeProviderIsCalledEachTime(t *testing.T) {
	var calls int
	ctx := platform.WithTimeProvider(context.Background(), func() time.Time {
		calls++
		return time.Unix(int64(calls), 0).UTC()
	})

	first := platform.Now(ctx)
	second := platform.Now(ctx)

	if first.Equal(second) {
		t.Errorf("Now() returned the same time %v on consecutive calls; provider should be invoked each call", first)
	}
	if calls != 2 {
		t.Errorf("provider called %d times, want 2", calls)
	}
}
