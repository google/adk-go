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

package optimize_test

import (
	"context"
	"testing"

	"google.golang.org/adk/optimize"
)

func TestGridSampler_IteratesAndExhausts(t *testing.T) {
	v1 := &optimize.Variant{ID: "v1"}
	v2 := &optimize.Variant{ID: "v2"}
	s := optimize.NewGridSampler("grid", []*optimize.Variant{v1, v2})
	got1, _ := s.Next(context.Background())
	got2, _ := s.Next(context.Background())
	if got1 != v1 || got2 != v2 {
		t.Errorf("order = %v %v", got1, got2)
	}
	_, err := s.Next(context.Background())
	if err != optimize.ErrSamplerExhausted {
		t.Errorf("err = %v, want ErrSamplerExhausted", err)
	}
}

func TestSearch_RunSortsBestFirst(t *testing.T) {
	v1 := &optimize.Variant{ID: "v1", Spec: 0.3}
	v2 := &optimize.Variant{ID: "v2", Spec: 0.9}
	v3 := &optimize.Variant{ID: "v3", Spec: 0.6}
	s := &optimize.Search{
		Sampler: optimize.NewGridSampler("g", []*optimize.Variant{v1, v2, v3}),
		Score: func(_ context.Context, v *optimize.Variant) (float64, map[string]any, error) {
			return v.Spec.(float64), nil, nil
		},
	}
	out, err := s.Run(context.Background())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("results = %d, want 3", len(out))
	}
	if out[0].Variant.ID != "v2" || out[1].Variant.ID != "v3" || out[2].Variant.ID != "v1" {
		t.Errorf("order: %v %v %v", out[0].Variant.ID, out[1].Variant.ID, out[2].Variant.ID)
	}
}

func TestSearch_StopOnFirst(t *testing.T) {
	v1 := &optimize.Variant{ID: "v1", Spec: 0.3}
	v2 := &optimize.Variant{ID: "v2", Spec: 0.9}
	v3 := &optimize.Variant{ID: "v3", Spec: 0.6}
	s := &optimize.Search{
		Sampler: optimize.NewGridSampler("g", []*optimize.Variant{v1, v2, v3}),
		Score: func(_ context.Context, v *optimize.Variant) (float64, map[string]any, error) {
			return v.Spec.(float64), nil, nil
		},
		StopOnFirst: true,
		StopScore:   0.8,
	}
	out, err := s.Run(context.Background())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(out) != 2 {
		t.Errorf("evaluated %d variants, want 2 (stop after v2)", len(out))
	}
}

func TestSearch_MaxVariants(t *testing.T) {
	variants := []*optimize.Variant{
		{ID: "v1"}, {ID: "v2"}, {ID: "v3"}, {ID: "v4"},
	}
	s := &optimize.Search{
		Sampler:     optimize.NewGridSampler("g", variants),
		Score:       func(_ context.Context, _ *optimize.Variant) (float64, map[string]any, error) { return 0.5, nil, nil },
		MaxVariants: 2,
	}
	out, _ := s.Run(context.Background())
	if len(out) != 2 {
		t.Errorf("evaluated %d, want 2", len(out))
	}
}

func TestSearch_RejectsNilSampler(t *testing.T) {
	s := &optimize.Search{Score: func(_ context.Context, _ *optimize.Variant) (float64, map[string]any, error) { return 0, nil, nil }}
	if _, err := s.Run(context.Background()); err == nil {
		t.Error("expected error")
	}
}

func TestSearch_RejectsNilScore(t *testing.T) {
	s := &optimize.Search{Sampler: optimize.NewGridSampler("g", nil)}
	if _, err := s.Run(context.Background()); err == nil {
		t.Error("expected error")
	}
}
