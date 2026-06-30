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

	"github.com/google/uuid"

	"google.golang.org/adk/v2/platform"
)

func TestNewUUIDDefaultIsRandomAndValid(t *testing.T) {
	first := platform.NewUUID(context.Background())
	second := platform.NewUUID(context.Background())

	if _, err := uuid.Parse(first); err != nil {
		t.Errorf("NewUUID() = %q, not a valid UUID: %v", first, err)
	}
	if first == second {
		t.Errorf("NewUUID() returned the same value %q twice; want random values", first)
	}
}

func TestNewUUIDNilContext(t *testing.T) {
	// NewUUID must tolerate a nil context and fall back to a random UUID.
	var nilCtx context.Context
	got := platform.NewUUID(nilCtx)
	if _, err := uuid.Parse(got); err != nil {
		t.Errorf("NewUUID(nil) = %q, not a valid UUID: %v", got, err)
	}
}

func TestWithUUIDProviderOverridesNewUUID(t *testing.T) {
	var n int
	ctx := platform.WithUUIDProvider(context.Background(), func() string {
		n++
		return "id-" + string(rune('0'+n))
	})

	want := []string{"id-1", "id-2", "id-3"}
	for i, w := range want {
		if got := platform.NewUUID(ctx); got != w {
			t.Errorf("NewUUID() call %d = %q, want %q", i+1, got, w)
		}
	}
}

func TestWithUUIDProviderDerivedContext(t *testing.T) {
	ctx := platform.WithUUIDProvider(context.Background(), func() string { return "fixed" })
	// TODO(kdroste): refactor underlying context
	derived, cancel := context.WithCancel(ctx)
	defer cancel()

	if got := platform.NewUUID(derived); got != "fixed" {
		t.Errorf("NewUUID() on derived context = %q, want %q", got, "fixed")
	}
}

func TestWithUUIDProviderNilFallsBack(t *testing.T) {
	ctx := platform.WithUUIDProvider(context.Background(), nil)

	got := platform.NewUUID(ctx)
	if _, err := uuid.Parse(got); err != nil {
		t.Errorf("NewUUID() with nil provider = %q, not a valid UUID: %v", got, err)
	}
}

func TestWithUUIDProviderNestedOverride(t *testing.T) {
	ctx := platform.WithUUIDProvider(context.Background(), func() string { return "outer" })
	ctx = platform.WithUUIDProvider(ctx, func() string { return "inner" })

	if got := platform.NewUUID(ctx); got != "inner" {
		t.Errorf("NewUUID() = %q, want innermost provider value %q", got, "inner")
	}
}
