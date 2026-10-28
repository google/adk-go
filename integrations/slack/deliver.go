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
	"log/slog"
	"slices"
	"strings"

	"google.golang.org/adk/integrations/slack/internal/slackapi"
)

// maxMessageRunes is Slack's recommended maximum length of a message's text field.
const maxMessageRunes = 4000

// deliverer posts one turn's replies into a Slack thread. Failures are logged
// and never stop the turn, because a lost reply is better than a lost turn.
type deliverer struct {
	api         *slackapi.Client
	channel     string
	threadTS    string
	placeholder string // "" = no placeholder
	errorText   string
	render      func(string) string // mrkdwn.Render or identity
	// logger is never nil. Callers with none configured pass
	// slog.New(slog.DiscardHandler).
	logger *slog.Logger

	placeholderTS string
	// replaced is set once a reply has been sent into the placeholder, so the
	// next reply is posted instead.
	replaced bool
	// stale is set when that update failed and the reply was posted instead,
	// so the placeholder still shows and finish must delete it.
	stale     bool
	delivered int
}

// start posts the placeholder.
func (d *deliverer) start(ctx context.Context) {
	if d.placeholder == "" {
		return
	}
	ts, err := d.api.PostMessage(ctx, d.channel, d.threadTS, d.placeholder)
	if err != nil {
		d.warn(ctx, "chat.postMessage", err)
		return
	}
	d.placeholderTS = ts
}

// text delivers one text part: the first replaces the placeholder, later
// ones are new thread replies.
func (d *deliverer) text(ctx context.Context, s string) {
	pieces := chunk(d.render(s), maxMessageRunes)
	ok := len(pieces) > 0
	for _, piece := range pieces {
		if !d.send(ctx, piece) {
			ok = false
		}
	}
	if ok {
		d.delivered++
	}
}

// finish reports turnErr in the thread and removes a placeholder that no
// reply replaced.
func (d *deliverer) finish(ctx context.Context, turnErr error) {
	if turnErr != nil {
		d.send(ctx, d.errorText)
	}
	if d.placeholderTS != "" && (!d.replaced || d.stale) {
		if err := d.api.DeleteMessage(ctx, d.channel, d.placeholderTS); err != nil {
			d.warn(ctx, "chat.delete", err)
		}
	}
}

// send puts s in the placeholder if it is still unused, else posts it. A
// failed update falls back to a post so the reply is not lost.
func (d *deliverer) send(ctx context.Context, s string) bool {
	if d.placeholderTS != "" && !d.replaced {
		d.replaced = true
		err := d.api.UpdateMessage(ctx, d.channel, d.placeholderTS, s)
		if err == nil {
			return true
		}
		d.warn(ctx, "chat.update", err)
		d.stale = true
	}
	if _, err := d.api.PostMessage(ctx, d.channel, d.threadTS, s); err != nil {
		d.warn(ctx, "chat.postMessage", err)
		return false
	}
	return true
}

// warn omits the message text, which may hold user data. WARN, not ERROR,
// because the turn carries on.
func (d *deliverer) warn(ctx context.Context, method string, err error) {
	d.logger.WarnContext(ctx, "slack: delivery failed", "channel", d.channel, "thread_ts", d.threadTS, "method", method, "error", err)
}

// chunk splits s into pieces of at most limit runes, cutting at the last
// paragraph break before the limit, else the last line break, else at the
// limit. The chosen break is dropped, so no piece ends or starts with that
// break, though a piece can still begin or end with another newline.
// Pieces with only whitespace are dropped, since Slack rejects a message
// with no visible text. text renders mrkdwn before calling chunk, so a split
// can fall inside a mrkdwn span such as *bold*.
func chunk(s string, limit int) []string {
	runes := []rune(s)
	var pieces []string
	add := func(r []rune) {
		if p := string(r); strings.TrimSpace(p) != "" {
			pieces = append(pieces, p)
		}
	}
	for len(runes) > limit {
		window := runes[:limit]
		cut, skip := limit, 0
		if i := lastIndex(window, []rune("\n\n")); i >= 0 {
			cut, skip = i, 2
		} else if i := lastIndex(window, []rune("\n")); i >= 0 {
			cut, skip = i, 1
		}
		add(runes[:cut])
		runes = runes[cut+skip:]
	}
	add(runes)
	return pieces
}

func lastIndex(s, sep []rune) int {
	for i := len(s) - len(sep); i >= 0; i-- {
		if slices.Equal(s[i:i+len(sep)], sep) {
			return i
		}
	}
	return -1
}
