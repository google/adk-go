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

// Package plugin provides lifecycle hooks that observe and intercept an agent
// run, exposing Before*/After* callbacks for the run, agent, model, and tool
// stages.
//
// An error a callback returns reaches the caller of the run wrapped as
// `Error in plugin '<plugin name>' during '<callback name>' callback: <error text>`,
// where the callback name is adk-python's (`before_run_callback`,
// `on_event_callback`, ...). The wrapper unwraps to the callback's error, so
// [errors.Is] and [errors.As] still find it; comparing with `==` does not.
// The tool callbacks (`BeforeToolCallback`, `AfterToolCallback`,
// `OnToolErrorCallback`) are not wrapped, and neither are the notification
// callbacks, which return nothing.
package plugin

import (
	"errors"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/session"
)

// Config configures a [Plugin], supplying its name and the lifecycle callbacks
// it registers for the run, agent, model, and tool stages.
type Config struct {
	Name string

	OnUserMessageCallback OnUserMessageCallback

	OnEventCallback OnEventCallback

	BeforeRunCallback BeforeRunCallback
	AfterRunCallback  AfterRunCallback
	// At most one of AfterRunCallback and FailableAfterRunCallback is set.
	FailableAfterRunCallback FailableAfterRunCallback
	OnRunErrorCallback       OnRunErrorCallback

	BeforeAgentCallback  agent.BeforeAgentCallback
	AfterAgentCallback   agent.AfterAgentCallback
	OnAgentErrorCallback OnAgentErrorCallback

	BeforeModelCallback  llmagent.BeforeModelCallback
	AfterModelCallback   llmagent.AfterModelCallback
	OnModelErrorCallback llmagent.OnModelErrorCallback

	BeforeToolCallback  llmagent.BeforeToolCallback
	AfterToolCallback   llmagent.AfterToolCallback
	OnToolErrorCallback llmagent.OnToolErrorCallback

	// CloseFunc releases the plugin's resources. runner.Runner.Close calls
	// it once per call, after the toolsets, and waits for it at most
	// runner.PluginConfig.CloseTimeout. CloseFunc should return: a call
	// that has not returned by then is abandoned, not canceled, and
	// keeps running in the background. Every call of Runner.Close calls
	// CloseFunc again, so an abandoned call may still be running when the
	// next one starts, and CloseFunc must be safe to call concurrently with
	// itself. A panic is recovered and reported as a failed close. A nil
	// CloseFunc means nothing to release.
	CloseFunc func() error
}

// New creates a [Plugin] from cfg. It returns an error when cfg sets both
// AfterRunCallback and FailableAfterRunCallback.
func New(cfg Config) (*Plugin, error) {
	if cfg.AfterRunCallback != nil && cfg.FailableAfterRunCallback != nil {
		return nil, errors.New("plugin: AfterRunCallback and FailableAfterRunCallback are both set")
	}
	p := &Plugin{
		name:                  cfg.Name,
		onUserMessageCallback: cfg.OnUserMessageCallback,
		onEventCallback:       cfg.OnEventCallback,
		beforeRunCallback:     cfg.BeforeRunCallback,
		afterRunCallback:      cfg.AfterRunCallback,
		failableAfterRun:      cfg.FailableAfterRunCallback,
		onRunErrorCallback:    cfg.OnRunErrorCallback,
		beforeAgentCallback:   cfg.BeforeAgentCallback,
		afterAgentCallback:    cfg.AfterAgentCallback,
		onAgentErrorCallback:  cfg.OnAgentErrorCallback,
		beforeModelCallback:   cfg.BeforeModelCallback,
		afterModelCallback:    cfg.AfterModelCallback,
		onModelErrorCallback:  cfg.OnModelErrorCallback,
		beforeToolCallback:    cfg.BeforeToolCallback,
		afterToolCallback:     cfg.AfterToolCallback,
		onToolErrorCallback:   cfg.OnToolErrorCallback,
		closeFunc:             cfg.CloseFunc,
	}

	// Ensure closeFunc is never nil so p.Close() doesn't panic
	if p.closeFunc == nil {
		p.closeFunc = func() error {
			return nil
		}
	}

	return p, nil
}

// Plugin holds a set of lifecycle callbacks that observe and intercept an agent
// run. Create one with [New].
type Plugin struct {
	name string

	onUserMessageCallback OnUserMessageCallback
	onEventCallback       OnEventCallback

	beforeRunCallback  BeforeRunCallback
	afterRunCallback   AfterRunCallback
	failableAfterRun   FailableAfterRunCallback
	onRunErrorCallback OnRunErrorCallback

	beforeAgentCallback agent.BeforeAgentCallback
	afterAgentCallback  agent.AfterAgentCallback

	onAgentErrorCallback OnAgentErrorCallback

	beforeModelCallback  llmagent.BeforeModelCallback
	afterModelCallback   llmagent.AfterModelCallback
	onModelErrorCallback llmagent.OnModelErrorCallback

	beforeToolCallback  llmagent.BeforeToolCallback
	afterToolCallback   llmagent.AfterToolCallback
	onToolErrorCallback llmagent.OnToolErrorCallback

	closeFunc func() error
}

// Name returns the name of the plugin.
func (p *Plugin) Name() string {
	return p.name
}

// Close safely calls the internal close function.
func (p *Plugin) Close() error {
	return p.closeFunc()
}

// --- Accessors ---

// OnUserMessageCallback returns the plugin's OnUserMessageCallback.
func (p *Plugin) OnUserMessageCallback() OnUserMessageCallback {
	return p.onUserMessageCallback
}

// OnEventCallback returns the plugin's OnEventCallback.
func (p *Plugin) OnEventCallback() OnEventCallback {
	return p.onEventCallback
}

// BeforeRunCallback returns the plugin's BeforeRunCallback.
func (p *Plugin) BeforeRunCallback() BeforeRunCallback {
	return p.beforeRunCallback
}

// AfterRunCallback returns the plugin's AfterRunCallback.
func (p *Plugin) AfterRunCallback() AfterRunCallback {
	return p.afterRunCallback
}

// FailableAfterRunCallback returns the plugin's FailableAfterRunCallback, or
// nil when the plugin does not set one.
func (p *Plugin) FailableAfterRunCallback() FailableAfterRunCallback {
	return p.failableAfterRun
}

// OnRunErrorCallback returns the plugin's OnRunErrorCallback, or nil when the
// plugin does not set one.
func (p *Plugin) OnRunErrorCallback() OnRunErrorCallback {
	return p.onRunErrorCallback
}

// BeforeAgentCallback returns the plugin's BeforeAgentCallback.
func (p *Plugin) BeforeAgentCallback() agent.BeforeAgentCallback {
	return p.beforeAgentCallback
}

// AfterAgentCallback returns the plugin's AfterAgentCallback.
func (p *Plugin) AfterAgentCallback() agent.AfterAgentCallback {
	return p.afterAgentCallback
}

// OnAgentErrorCallback returns the plugin's OnAgentErrorCallback.
func (p *Plugin) OnAgentErrorCallback() OnAgentErrorCallback {
	return p.onAgentErrorCallback
}

// BeforeModelCallback returns the plugin's BeforeModelCallback.
func (p *Plugin) BeforeModelCallback() llmagent.BeforeModelCallback {
	return p.beforeModelCallback
}

// AfterModelCallback returns the plugin's AfterModelCallback.
func (p *Plugin) AfterModelCallback() llmagent.AfterModelCallback {
	return p.afterModelCallback
}

// OnModelErrorCallback returns the plugin's OnModelErrorCallback.
func (p *Plugin) OnModelErrorCallback() llmagent.OnModelErrorCallback {
	return p.onModelErrorCallback
}

// BeforeToolCallback returns the plugin's BeforeToolCallback.
func (p *Plugin) BeforeToolCallback() llmagent.BeforeToolCallback {
	return p.beforeToolCallback
}

// AfterToolCallback returns the plugin's AfterToolCallback.
func (p *Plugin) AfterToolCallback() llmagent.AfterToolCallback {
	return p.afterToolCallback
}

// OnToolErrorCallback returns the plugin's OnToolErrorCallback.
func (p *Plugin) OnToolErrorCallback() llmagent.OnToolErrorCallback {
	return p.onToolErrorCallback
}

// OnUserMessageCallback is invoked when the user sends a message, and may
// return a replacement message.
type OnUserMessageCallback func(agent.InvocationContext, *genai.Content) (*genai.Content, error)

// BeforeRunCallback is invoked before a run starts, and may return content that
// short-circuits the run.
//
// The runner calls the plugins' BeforeRunCallback in registration order after
// it has stored the run's user event and before any agent or node runs. The
// first callback that returns non-nil content ends the chain: the runner runs
// no agent or node, wraps the content in an event authored by the agent the
// turn would have run (or "model" on a runner.Config.Node root), passes that
// event through the OnEventCallback plugins, stamps the "adk:before_run_reply"
// custom metadata key on it, appends it to the session unless it is partial,
// and yields it; apart from the user event that runner.WithYieldUserMessage
// yields, the run yields nothing else. A callback error, wrapped as described
// in the package comment, ends the run. In runner.Runner.Run the after-run
// callbacks run in either case.
type BeforeRunCallback func(agent.InvocationContext) (*genai.Content, error)

// AfterRunCallback is invoked once at the end of a run that did not end
// because of a run error: a completed run, a run whose caller kept ranging
// past an error and that then ended without another, a run whose caller
// stopped ranging at an event, and a run whose context was canceled. It is
// not called when a run error ends the run (see [OnRunErrorCallback]), so
// cleanup that must also happen on a failed run
// belongs in OnRunErrorCallback too. The exception is a run that failed in
// a BeforeRunCallback or while handling its reply: there OnRunErrorCallback
// is called and the after-run callbacks still run.
//
// At the normal end of a run it is called after the last event and before
// post-invocation compaction, so the compaction summary event is not yet in
// the session. Plugins are called in registration order. A panic is not
// recovered. A plugin whose cleanup can fail uses [FailableAfterRunCallback]
// instead.
type AfterRunCallback func(agent.InvocationContext)

// FailableAfterRunCallback is an after-run callback that can report failure.
// It is called on the same runs as [AfterRunCallback] and at the same point,
// including before post-invocation compaction. A plugin sets at most one of
// the two: [New] rejects a [Config] that sets both. A nil return means
// success and changes nothing.
//
// A non-nil error is a run error and ends the run. The runner yields it,
// wrapped as described in the package comment, to a caller that is still
// ranging, and calls every plugin's [OnRunErrorCallback] with it once. When the caller has already
// stopped ranging the error cannot be yielded: it goes only to
// OnRunErrorCallback, and is recorded as an exception event on the
// invocation span when that span is recording. With no plugin implementing
// OnRunErrorCallback and no recording span it is not reported anywhere, so a
// plugin that must not lose its own failure handles it before returning.
//
// Plugins are called in registration order, and after the first plugin
// whose after-run callback fails, the after-run callbacks of later plugins
// are not called, as in adk-python. A panic is not recovered. The runner
// calls it at most once per run, on the goroutine that ranges the run; a
// plugin is shared by concurrent runs, so the function must be safe for
// concurrent use.
type FailableAfterRunCallback func(ctx agent.InvocationContext) error

// OnRunErrorCallback is invoked when a run ends because of a run error, with
// that error. It is notification-only, like [OnAgentErrorCallback]: it cannot
// change the outcome, and the caller of the run receives err unchanged.
//
// A run error is an error raised once the run has started: by the agent or
// workflow node, by a run-level plugin callback (BeforeRunCallback,
// OnEventCallback, OnUserMessageCallback, a failing
// [FailableAfterRunCallback]), by appending an event to the session, or by
// post-invocation compaction. A run ends because of a run error when the
// caller stops ranging at a yielded run error, or when the runner itself
// stops on one. The callback is then called at most once per run, for every
// plugin in registration order, after the OnAgentErrorCallback calls for the
// same error, and err is the value the caller received: the runner does not
// wrap the agent's or node's error. For a caller that stops at the error, the
// call happens after the caller's loop body has returned and before its range
// statement completes. A caller that keeps ranging past an error at a place
// where the run goes on does not end the run with it: the callback is not
// called for that error, and if the run later ends because of another run
// error, it is called once for that one.
//
// It is not called for a failure before the run starts (session lookup,
// message checks), for the cancellation of the invocation's context, when
// the caller stops ranging at an event, or for a panic.
// [AfterRunCallback] is not called on a run that ended because of a run
// error, except for an error from a BeforeRunCallback or from handling its
// reply. On the live path, an error that Runner.RunLive returns is notified
// before RunLive returns.
//
// A panicking callback is recovered, so the remaining plugins are still
// notified and the caller still holds the original error. The panic is
// recorded as an exception event on the invocation span when that span is
// recording, and is not reported anywhere else; a plugin that must see its
// own failures recovers them itself.
//
// Within a run the runner calls it at most once, on the goroutine that
// ranges the run. A plugin is shared by concurrent runs, so the function must
// be safe for concurrent use. The runner does not serialize it against
// OnAgentErrorCallback, which parallel sub-agents may call concurrently, so a
// plugin whose two hooks share state guards that state with its own lock.
type OnRunErrorCallback func(ctx agent.InvocationContext, err error)

// OnAgentErrorCallback is invoked when an error escapes an agent's run. It is
// notification-only: callbacks are called in registration order and the error
// is yielded to the caller unchanged.
//
// a is the agent that failed. It may be the embedded agent implementation
// rather than the value returned by a constructor, so use a.Name() to identify
// it. ctx is the callback context for the invocation, and ctx.Agent() returns
// a. State changes made through ctx are not persisted. Artifact saves through ctx.Artifacts() are written to the artifact
// service, but no ArtifactDelta records them.
//
// A panicking callback is recovered so the remaining callbacks still run and
// the original error is still yielded, matching adk-python's
// on_agent_error_callback. An error is compared against the failing agent's
// own context: when that context is done, a cancellation or deadline error is
// not reported, matching adk-python's handling of task cancellation — and that
// covers a deadline a parent agent or a workflow node imposed on it, since
// that deadline lives on the agent's own context. The context's cancellation
// cause, or any error wrapping it, is suppressed too, because errors.Is
// cannot tell an echo of the cancellation from an unrelated failure that
// wraps it; a caller that cancels with a cause should use an error value it
// owns. Errors that escape while the agent's context is still live are
// reported. Callbacks may
// be invoked concurrently by parallel sub-agents and must be safe for
// concurrent use. An error from a sub-agent is reported once for each ancestor
// as it propagates, matching adk-python; callbacks that count failures should
// dedupe with errors.Is because ancestors wrap the child's error. RunLive
// bypasses the non-live run loop, so it does not invoke this callback.
//
// When the error then ends the run, every plugin's [OnRunErrorCallback] is
// called after the OnAgentErrorCallback calls for it. The runner does not
// serialize the two hooks against each other, so a plugin whose hooks share
// state guards that state with its own lock.
type OnAgentErrorCallback func(ctx agent.Context, a agent.Agent, err error)

// OnEventCallback is invoked for each event before the runner decides whether
// to persist it to the session service and yields it to the caller.
//
// Callbacks run in plugin registration order. A callback may either mutate the
// incoming *session.Event in place and return (nil, nil) so later plugins still
// run, or return a non-nil *session.Event, which stops the callback chain.
//
// When a callback returns a new *session.Event pointer, the runner merges the
// fields set on that replacement onto a copy of the original event rather than
// swapping the struct whole:
//   - ID, InvocationID, and Timestamp always come from the original event.
//   - Author takes the replacement's Author if non-empty, and otherwise falls
//     back to the original event's Author.
//   - Actions.Compaction always keeps the framework's compaction record from
//     the original event.
//   - Partial takes the replacement's Partial when Partial is true or
//     PartialSet is true (see [google.golang.org/adk/v2/model.LLMResponse.PartialSet]).
//     A replacement, whether a struct literal or a copy of the original event,
//     with Partial: false and PartialSet: false leaves the original's Partial
//     in place. To downgrade a partial event to non-partial, set Partial = false
//     on the incoming event and return nil or the same pointer, or return a
//     replacement with Partial: false and PartialSet: true.
//   - CustomMetadata, when nil on the replacement, keeps the original event's
//     CustomMetadata (with any non-internal RunConfig.CustomMetadata entries
//     merged underneath). When non-nil on the replacement (including an empty
//     map[string]any{} to clear public keys), the replacement's entries
//     replace the original event's public keys while preserving any
//     ADK-internal keys ("__adk_internal_*", "adk:before_run_reply",
//     "adk.workflow.*") from the original event and merging
//     RunConfig.CustomMetadata underneath.
//   - Every other field on session.Event, model.LLMResponse, and
//     session.EventActions overrides the original event's value when non-zero
//     on the replacement, and keeps the original event's value when left at
//     its zero value. To clear a non-zero field on the original event (other
//     than Partial via PartialSet or CustomMetadata via map[string]any{}),
//     mutate the incoming event in place (and return nil or the same pointer).
//
// Neither the original event's maps nor the replacement event's maps are
// mutated by the runner's merge.
type OnEventCallback func(agent.InvocationContext, *session.Event) (*session.Event, error)
