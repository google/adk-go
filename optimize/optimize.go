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

// Package optimize provides minimal scaffolding for prompt / agent
// optimization. Mirrors a small subset of adk-python's
// google.adk.optimization.
//
// Phase 10 ships the core data shapes — Variant, Sampler, ScoreFunc,
// Search — plus a deterministic GridSearch implementation. LLM-based
// samplers and the full evaluator integration ship in dedicated
// subpackages.
package optimize

import (
	"context"
	"errors"
	"sort"
)

// Variant is a candidate configuration produced by a Sampler. Concrete
// fields are user-defined; the engine only requires that variants be
// scoreable and identifiable.
type Variant struct {
	// ID is a stable identifier for telemetry and reporting.
	ID string

	// Spec is the user-defined configuration the sampler produced. The
	// ScoreFunc is responsible for interpreting it.
	Spec any

	// Description is optional human-readable context.
	Description string
}

// Sampler produces variants to evaluate. Implementations may be
// deterministic (grid search) or stochastic (LLM-driven proposal).
type Sampler interface {
	Name() string
	Next(ctx context.Context) (*Variant, error)
}

// ErrSamplerExhausted signals that a Sampler has produced its final
// variant. Search loops stop on this error.
var ErrSamplerExhausted = errors.New("optimize: sampler exhausted")

// ScoreFunc evaluates a variant and returns a score in [0, 1] (higher
// is better) plus optional metadata.
type ScoreFunc func(ctx context.Context, v *Variant) (score float64, meta map[string]any, err error)

// Result is one entry in a Search's output, pairing a variant with its
// score and any metadata the ScoreFunc returned.
type Result struct {
	Variant *Variant
	Score   float64
	Meta    map[string]any
	Err     error
}

// Search drives a Sampler to completion, scores each variant, and
// returns the results sorted by score (descending).
//
// The MaxVariants field bounds the number of variants evaluated; 0
// means evaluate everything the sampler produces.
type Search struct {
	Sampler      Sampler
	Score        ScoreFunc
	MaxVariants  int
	StopOnFirst  bool // when true, stop as soon as a variant scores >= StopScore
	StopScore    float64
}

// Run executes the search and returns results sorted best-first.
func (s *Search) Run(ctx context.Context) ([]Result, error) {
	if s.Sampler == nil {
		return nil, errors.New("optimize: Search.Sampler is nil")
	}
	if s.Score == nil {
		return nil, errors.New("optimize: Search.Score is nil")
	}
	var out []Result
	for s.MaxVariants == 0 || len(out) < s.MaxVariants {
		v, err := s.Sampler.Next(ctx)
		if errors.Is(err, ErrSamplerExhausted) {
			break
		}
		if err != nil {
			return nil, err
		}
		score, meta, scoreErr := s.Score(ctx, v)
		out = append(out, Result{Variant: v, Score: score, Meta: meta, Err: scoreErr})
		if s.StopOnFirst && scoreErr == nil && score >= s.StopScore {
			break
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	return out, nil
}

// GridSampler iterates over a slice of pre-built variants. Useful for
// classical grid / template search.
type GridSampler struct {
	name     string
	variants []*Variant
	idx      int
}

// NewGridSampler constructs a GridSampler from a fixed list.
func NewGridSampler(name string, variants []*Variant) *GridSampler {
	return &GridSampler{name: name, variants: variants}
}

// Name implements Sampler.
func (g *GridSampler) Name() string { return g.name }

// Next implements Sampler. Returns ErrSamplerExhausted when the slice
// is fully consumed.
func (g *GridSampler) Next(_ context.Context) (*Variant, error) {
	if g.idx >= len(g.variants) {
		return nil, ErrSamplerExhausted
	}
	v := g.variants[g.idx]
	g.idx++
	return v, nil
}
