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

// Package contextfilter provides a plugin that shortens the history sent to
// the model: it keeps only the latest invocations of the conversation and can
// pass the result through a caller-supplied filter.
//
// Register the plugin with a runner, and it runs before every model call of
// every agent of that runner:
//
//	p, err := contextfilter.New(contextfilter.Config{NumInvocationsToKeep: 3})
//	if err != nil {
//		return err
//	}
//	r, err := runner.New(runner.Config{
//		AppName:        "app",
//		Agent:          rootAgent,
//		SessionService: session.InMemoryService(),
//		PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
//	})
//
// The plugin changes only the contents of the request sent to the model. The
// session keeps every event, so each model call trims the whole history again.
package contextfilter

import (
	"errors"
	"fmt"
	"slices"

	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
)

const defaultName = "context_filter_plugin"

// The scan of the contents meets a nil entry where adk-python reads an
// attribute of it and raises; the trim then fails like a failing filter.
var (
	errNilContent = errors.New("request holds a nil content")
	errNilPart    = errors.New("request holds a nil part")
)

// Config configures a context filter plugin. The zero value is valid: no
// trimming, no custom filter, plugin name "context_filter_plugin".
type Config struct {
	// Name is the plugin's name, unique among the plugins of one runner.
	// Empty selects "context_filter_plugin".
	Name string

	// NumInvocationsToKeep is how many of the latest invocations each model
	// request keeps. Zero or negative keeps the whole history.
	//
	// An invocation starts at a content whose role is "user" and that holds no
	// function response, when the content before it is not such a content.
	// Contents are counted by role only, so another agent's output that the
	// flow rewrites into a "For context:" user content also starts an
	// invocation.
	//
	// When trimming applies and the request holds a nil content, or a nil part
	// that the scan reaches, the plugin leaves the request unchanged.
	NumInvocationsToKeep int

	// RemoveAmount delays trimming until the request holds at least
	// NumInvocationsToKeep+RemoveAmount invocations. Nil selects 1; a value
	// below 1 makes New fail. New reads the value once.
	//
	// Once the threshold is reached, every later request is trimmed to exactly
	// NumInvocationsToKeep invocations, so the kept window moves by one
	// invocation per turn; it is not a batched window.
	RemoveAmount *int

	// CustomFilter, when non-nil, receives the contents that trimming left
	// (the whole history when trimming is off or below its threshold) and
	// returns the contents the model receives. It runs on every model call, at
	// most once per request, on the goroutine that makes the model call, also
	// for a request with no contents.
	//
	// The slice it receives is a copy of the request's slice; its elements are
	// the request's own *genai.Content values. It may return its argument, a
	// sub-slice or a new slice. An empty or nil result with a nil error is
	// valid and gives the model a request with no contents.
	//
	// A non-nil error, or a panic, leaves the request exactly as it was before
	// the plugin ran, a successful trim included, and the run continues. The
	// failure is recorded as an error event on the current span. The filter
	// must be safe for concurrent use.
	CustomFilter func(contents []*genai.Content) ([]*genai.Content, error)
}

// New returns a plugin whose before-model callback shortens the model
// request's contents as Config describes and then applies CustomFilter. The
// callback never returns a response or an error, so the model is always
// called. It changes only the request's Contents. cfg is read once and not
// retained.
//
// New fails, returning a nil plugin, when RemoveAmount is non-nil and below 1,
// even if NumInvocationsToKeep is not positive. The returned plugin is
// immutable and safe for concurrent use if cfg.CustomFilter is.
func New(cfg Config) (*plugin.Plugin, error) {
	name := cfg.Name
	if name == "" {
		name = defaultName
	}
	removeAmount := 1
	if cfg.RemoveAmount != nil {
		if *cfg.RemoveAmount < 1 {
			return nil, fmt.Errorf("contextfilter: RemoveAmount must be at least 1, got %d", *cfg.RemoveAmount)
		}
		removeAmount = *cfg.RemoveAmount
	}
	f := &filter{
		name:         name,
		keep:         cfg.NumInvocationsToKeep,
		removeAmount: removeAmount,
		custom:       cfg.CustomFilter,
	}
	return plugin.New(plugin.Config{
		Name:                name,
		BeforeModelCallback: f.beforeModel,
	})
}

// filter holds a plugin's configuration. It is never modified after New, so
// one value serves concurrent model calls.
type filter struct {
	name         string
	keep         int
	removeAmount int
	custom       func([]*genai.Content) ([]*genai.Content, error)
}

// beforeModel replaces req.Contents with the result of reduce. A failure
// leaves req untouched and is recorded on the current span instead of being
// returned, because a returned error would stop the model call.
func (f *filter) beforeModel(ctx agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	out, err := f.reduce(req.Contents)
	if err != nil {
		trace.SpanFromContext(ctx).RecordError(fmt.Errorf("plugin %q: %w", f.name, err))
		return nil, nil
	}
	req.Contents = out
	return nil, nil
}

// reduce trims contents and then applies the custom filter. The filter gets a
// copy of the slice, so a filter that writes to its argument and then fails
// cannot alter the request.
func (f *filter) reduce(contents []*genai.Content) (out []*genai.Content, err error) {
	if f.keep > 0 {
		contents, err = trim(contents, f.keep, f.removeAmount)
		if err != nil {
			return nil, err
		}
	}
	if f.custom == nil {
		return contents, nil
	}
	defer func() {
		if r := recover(); r != nil {
			out, err = nil, fmt.Errorf("custom filter panicked: %v", r)
		}
	}()
	out, err = f.custom(slices.Clone(contents))
	if err != nil {
		return nil, fmt.Errorf("custom filter failed: %w", err)
	}
	return out, nil
}

// trim returns the tail of contents that holds the last keep invocations,
// once contents hold at least keep+removeAmount of them; otherwise contents.
// keep and removeAmount are positive.
func trim(contents []*genai.Content, keep, removeAmount int) ([]*genai.Content, error) {
	starts, err := invocationStarts(contents)
	if err != nil {
		return nil, err
	}
	// len(starts) >= keep+removeAmount, written so that it cannot overflow.
	if len(starts)-keep < removeAmount {
		return contents, nil
	}
	split, err := extendSplitOverPairs(contents, starts[len(starts)-keep])
	if err != nil {
		return nil, err
	}
	return contents[split:], nil
}

// invocationStarts returns the index of each human user content whose
// predecessor is not one.
func invocationStarts(contents []*genai.Content) ([]int, error) {
	var starts []int
	prevHuman := false
	for i, c := range contents {
		human, err := isHumanUser(c)
		if err != nil {
			return nil, err
		}
		if human && !prevHuman {
			starts = append(starts, i)
		}
		prevHuman = human
	}
	return starts, nil
}

// isHumanUser reports whether c has the role "user" and no function response
// part. Its parts are read up to the first function response only, as
// adk-python's any() does.
func isHumanUser(c *genai.Content) (bool, error) {
	if c == nil {
		return false, errNilContent
	}
	if c.Role != genai.RoleUser {
		return false, nil
	}
	for _, p := range c.Parts {
		if p == nil {
			return false, errNilPart
		}
		if p.FunctionResponse != nil {
			return false, nil
		}
	}
	return true, nil
}

// extendSplitOverPairs returns the largest index i <= split such that every
// function response in contents[i:] with a non-empty ID has its function call
// in contents[i:] too, or 0 if there is none. contents hold no nil content:
// invocationStarts has already rejected one.
func extendSplitOverPairs(contents []*genai.Content, split int) (int, error) {
	needed := map[string]struct{}{}
	for i, content := range slices.Backward(contents) {
		parts := content.Parts
		for _, p := range slices.Backward(parts) {
			if p == nil {
				return 0, errNilPart
			}
			if p.FunctionResponse != nil && p.FunctionResponse.ID != "" {
				needed[p.FunctionResponse.ID] = struct{}{}
			}
			if p.FunctionCall != nil && p.FunctionCall.ID != "" {
				delete(needed, p.FunctionCall.ID)
			}
		}
		if i <= split && len(needed) == 0 {
			return i, nil
		}
	}
	return 0, nil
}
