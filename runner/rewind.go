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

package runner

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"reflect"
	"slices"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
)

// ErrInvocationNotFound is returned by [Runner.Rewind] when the target
// invocation ID is empty or names no event among the session's loaded events.
// Match it with [errors.Is].
var ErrInvocationNotFound = errors.New("invocation ID not found")

// Rewind appends a rewind event to the session, restoring its state and
// artifacts to what they were before the first event of the invocation
// rewindBeforeInvocationID. It runs no agent and no plugin callback.
//
// The session is looked up as [Runner.Run] looks it up: cfg limits the events
// loaded (nil loads all of them), and when the runner was created with
// [Config.AutoCreateSession] a missing session is created first. Only the
// loaded events are searched and folded, so a window that excludes the
// target invocation makes Rewind fail, and keys set before the window are
// treated as absent at the rewind point.
//
// The appended event has author "user", an invocation ID of the form
// "e-<uuid>", no content, rewindBeforeInvocationID in
// [session.EventActions.RewindBeforeInvocationID], and two deltas:
//
//   - The state delta restores every session-scoped key to its value at the
//     rewind point, as folded from the state deltas of the earlier events.
//     A key that is in the session state now but in no earlier event's
//     delta, including one set when the session was created, is set to nil.
//     Keys with the [session.KeyPrefixApp], [session.KeyPrefixUser] and
//     [session.KeyPrefixTemp] prefixes are left alone, and so are keys whose
//     value is unchanged ([reflect.DeepEqual]). The in-memory session service
//     then holds a key set to nil with a nil value; the Vertex AI session
//     service drops nil values when it reads a session back, so such a key
//     reads back absent there.
//   - The artifact delta covers session-scoped artifacts whose version
//     changed after the rewind point. Each is saved again as a new version:
//     the content of the version recorded before the rewind point, loaded
//     through the runner's artifact service, or [artifact.NewRewindTombstone]
//     when the artifact did not exist then or that version no longer loads.
//     The delta records the version Save returns. Artifacts whose name has
//     the "user:" prefix are left alone. A runner without an artifact
//     service rewinds state only.
//
// Rewind returns an error wrapping [ErrInvocationNotFound] when
// rewindBeforeInvocationID is empty, which it reports before reading the
// session, or names no loaded event; nothing is saved or appended then. A
// missing session without auto-create gives the session service's error,
// which wraps [session.ErrNotFound]. Artifact and session service errors are
// returned wrapped.
//
// The artifacts are saved before the event is appended, and the two services
// share no transaction: when a save or the append fails, the artifacts saved
// so far stay rewound and the session does not. For the same reason,
// concurrent rewinds of one session are not supported.
func (r *Runner) Rewind(ctx context.Context, userID, sessionID, rewindBeforeInvocationID string, cfg *agent.GetSessionConfig) error {
	// adk-python accepts an empty ID, which matches any event without an
	// invocation ID and so can rewind the whole session.
	if rewindBeforeInvocationID == "" {
		return fmt.Errorf("%w: rewind_before_invocation_id must be non-empty", ErrInvocationNotFound)
	}
	sess, err := r.getOrCreateSession(ctx, userID, sessionID, getSessionConfigOf(cfg))
	if err != nil {
		return err
	}
	events := slices.Collect(sess.Events().All())
	rewindIndex := slices.IndexFunc(events, func(ev *session.Event) bool {
		return ev.InvocationID == rewindBeforeInvocationID
	})
	if rewindIndex < 0 {
		return fmt.Errorf("%w: %s", ErrInvocationNotFound, rewindBeforeInvocationID)
	}

	stateDelta := rewindStateDelta(sess.State(), events[:rewindIndex])
	artifactDelta, err := r.rewindArtifacts(ctx, sess, events, rewindIndex)
	if err != nil {
		return err
	}

	ev := session.NewEvent(ctx, "e-"+platform.NewUUID(ctx))
	ev.Author = "user"
	ev.Actions.RewindBeforeInvocationID = rewindBeforeInvocationID
	ev.Actions.StateDelta = stateDelta
	ev.Actions.ArtifactDelta = artifactDelta
	if err := r.sessionService.AppendEvent(ctx, sess, ev); err != nil {
		return fmt.Errorf("rewind: append rewind event: %w", err)
	}
	return nil
}

// isSessionScopedKey reports whether a rewind restores the state key.
func isSessionScopedKey(key string) bool {
	return !strings.HasPrefix(key, session.KeyPrefixApp) &&
		!strings.HasPrefix(key, session.KeyPrefixUser) &&
		!strings.HasPrefix(key, session.KeyPrefixTemp)
}

// rewindStateDelta returns the delta that turns current into the
// session-scoped state folded from the state deltas of before. It ports
// adk-python's compute_state_delta_for_rewind (sessions/_rewind_utils.py).
func rewindStateDelta(current session.State, before []*session.Event) map[string]any {
	atRewind := map[string]any{}
	for _, ev := range before {
		for key, value := range ev.Actions.StateDelta {
			if !isSessionScopedKey(key) {
				continue
			}
			if value == nil {
				delete(atRewind, key)
			} else {
				atRewind[key] = value
			}
		}
	}

	currentState := maps.Collect(current.All())
	delta := map[string]any{}
	for key, value := range atRewind {
		// State values are often maps and slices, which == cannot compare.
		if now, ok := currentState[key]; !ok || !reflect.DeepEqual(now, value) {
			delta[key] = value
		}
	}
	for key := range currentState {
		if _, ok := atRewind[key]; !ok && isSessionScopedKey(key) {
			delta[key] = nil
		}
	}
	return delta
}

// rewindArtifacts saves every session-scoped artifact whose version changed
// at or after events[rewindIndex] again with its content at the rewind point,
// and returns the versions saved. It ports adk-python's
// compute_artifact_delta_for_rewind (sessions/_rewind_utils.py).
func (r *Runner) rewindArtifacts(ctx context.Context, sess session.Session, events []*session.Event, rewindIndex int) (map[string]int64, error) {
	delta := map[string]int64{}
	if r.artifactService == nil {
		return delta, nil
	}

	atRewind := map[string]int64{}
	current := map[string]int64{}
	for i, ev := range events {
		if i < rewindIndex {
			maps.Copy(atRewind, ev.Actions.ArtifactDelta)
		}
		maps.Copy(current, ev.Actions.ArtifactDelta)
	}

	// Sorted, so that a failing save leaves the same artifacts rewound every
	// time.
	for _, fileName := range slices.Sorted(maps.Keys(current)) {
		if strings.HasPrefix(fileName, "user:") {
			continue
		}
		version, existed := atRewind[fileName]
		if existed && version == current[fileName] {
			continue
		}
		part := artifact.NewRewindTombstone()
		if existed {
			loaded, err := r.loadArtifactVersion(ctx, sess, fileName, version)
			if err != nil {
				return nil, err
			}
			if loaded != nil {
				part = loaded
			}
		}
		resp, err := r.artifactService.Save(ctx, &artifact.SaveRequest{
			AppName:   sess.AppName(),
			UserID:    sess.UserID(),
			SessionID: sess.ID(),
			FileName:  fileName,
			Part:      part,
		})
		if err != nil {
			return nil, fmt.Errorf("rewind: save artifact: %w", err)
		}
		delta[fileName] = resp.Version
	}
	return delta, nil
}

// loadArtifactVersion returns the content of one artifact version, or nil
// when it does not exist. adk-python logs a warning in that case; adk-go
// library code does not log, and the caller saves a tombstone either way.
func (r *Runner) loadArtifactVersion(ctx context.Context, sess session.Session, fileName string, version int64) (*genai.Part, error) {
	resp, err := r.artifactService.Load(ctx, &artifact.LoadRequest{
		AppName:   sess.AppName(),
		UserID:    sess.UserID(),
		SessionID: sess.ID(),
		FileName:  fileName,
		Version:   version,
	})
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("rewind: load artifact version %d: %w", version, err)
	}
	return resp.Part, nil
}
