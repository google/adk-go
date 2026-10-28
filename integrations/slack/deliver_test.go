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
	"bytes"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"google.golang.org/adk/integrations/slack/internal/slackapi"
	"google.golang.org/adk/integrations/slack/internal/slackapitest"
	"google.golang.org/adk/integrations/slack/mrkdwn"
)

const (
	testChannel = "C1"
	testThread  = "1700000000.000100"
	testError   = "Sorry, something went wrong."
)

func newTestDeliverer(f *slackapitest.Fake, placeholder string) *deliverer {
	return &deliverer{
		api:         slackapi.New("xoxb-test", slackapi.WithBaseURL(f.URL())),
		channel:     testChannel,
		threadTS:    testThread,
		placeholder: placeholder,
		errorText:   testError,
		render:      func(s string) string { return s },
		logger:      slog.New(slog.DiscardHandler),
	}
}

// The fake numbers posted messages in order, so the n-th successful post has ts(n).
func ts(n int) string { return fmt.Sprintf("1000000000.%06d", n) }

func post(n int, text string) slackapitest.Call {
	return slackapitest.Call{Method: "chat.postMessage", Channel: testChannel, ThreadTS: testThread, TS: ts(n), Text: text}
}

func update(n int, text string) slackapitest.Call {
	return slackapitest.Call{Method: "chat.update", Channel: testChannel, TS: ts(n), Text: text}
}

func del(n int) slackapitest.Call {
	return slackapitest.Call{Method: "chat.delete", Channel: testChannel, TS: ts(n)}
}

func methods(calls []slackapitest.Call) []string {
	m := make([]string, len(calls))
	for i, c := range calls {
		m[i] = c.Method
	}
	return m
}

func TestDelivererSequences(t *testing.T) {
	turnErr := errors.New("model failed")
	updateFails := map[string][]slackapitest.Response{"chat.update": {{Status: http.StatusInternalServerError}}}
	tests := []struct {
		name        string
		placeholder string
		fail        map[string][]slackapitest.Response
		parts       []string
		err         error
		want        []slackapitest.Call
	}{
		{
			name:        "one part replaces the placeholder",
			placeholder: "Thinking…",
			parts:       []string{"one"},
			want:        []slackapitest.Call{post(1, "Thinking…"), update(1, "one")},
		},
		{
			name:        "later parts are new thread replies",
			placeholder: "Thinking…",
			parts:       []string{"one", "two", "three"},
			want:        []slackapitest.Call{post(1, "Thinking…"), update(1, "one"), post(2, "two"), post(3, "three")},
		},
		{
			name:        "no parts deletes the placeholder",
			placeholder: "Thinking…",
			want:        []slackapitest.Call{post(1, "Thinking…"), del(1)},
		},
		{
			name:        "error after a part is a new reply",
			placeholder: "Thinking…",
			parts:       []string{"one"},
			err:         turnErr,
			want:        []slackapitest.Call{post(1, "Thinking…"), update(1, "one"), post(2, testError)},
		},
		{
			name:        "error with no parts replaces the placeholder",
			placeholder: "Thinking…",
			err:         turnErr,
			want:        []slackapitest.Call{post(1, "Thinking…"), update(1, testError)},
		},
		{
			name:        "failed update posts the part and deletes the stale placeholder",
			placeholder: "Thinking…",
			fail:        updateFails,
			parts:       []string{"one"},
			want:        []slackapitest.Call{post(1, "Thinking…"), update(1, "one"), post(2, "one"), del(1)},
		},
		{
			name:        "failed update, then a second part",
			placeholder: "Thinking…",
			fail:        updateFails,
			parts:       []string{"one", "two"},
			want:        []slackapitest.Call{post(1, "Thinking…"), update(1, "one"), post(2, "one"), post(3, "two"), del(1)},
		},
		{
			name:        "failed update, then a turn error",
			placeholder: "Thinking…",
			fail:        updateFails,
			parts:       []string{"one"},
			err:         turnErr,
			want:        []slackapitest.Call{post(1, "Thinking…"), update(1, "one"), post(2, "one"), post(3, testError), del(1)},
		},
		{
			name:  "no placeholder, one part",
			parts: []string{"one"},
			want:  []slackapitest.Call{post(1, "one")},
		},
		{
			name: "no placeholder, no parts",
			want: []slackapitest.Call{},
		},
		{
			name: "no placeholder, error",
			err:  turnErr,
			want: []slackapitest.Call{post(1, testError)},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := slackapitest.New(t)
			// The fake consumes its queues in place, so each row gets its own map.
			f.Fail = maps.Clone(tt.fail)
			d := newTestDeliverer(f, tt.placeholder)
			d.start(t.Context())
			for _, p := range tt.parts {
				d.text(t.Context(), p)
			}
			d.finish(t.Context(), tt.err)

			got := f.Snapshot()
			if !slices.Equal(f.Methods(), methods(tt.want)) {
				t.Fatalf("methods = %q, want %q", f.Methods(), methods(tt.want))
			}
			if !slices.Equal(got, tt.want) {
				t.Errorf("calls =\n%+v\nwant\n%+v", got, tt.want)
			}
			if d.delivered != len(tt.parts) {
				t.Errorf("delivered = %d, want %d", d.delivered, len(tt.parts))
			}
		})
	}
}

func TestDelivererContinuesAfterFailedPost(t *testing.T) {
	f := slackapitest.New(t)
	d := newTestDeliverer(f, "Thinking…")
	d.start(t.Context())
	d.text(t.Context(), "one")
	// Scripted only now so the placeholder post is not the one that fails.
	f.Fail = map[string][]slackapitest.Response{"chat.postMessage": {{Status: http.StatusInternalServerError}}}
	d.text(t.Context(), "two")
	d.text(t.Context(), "three")
	d.finish(t.Context(), nil)

	failed := slackapitest.Call{Method: "chat.postMessage", Channel: testChannel, ThreadTS: testThread, Text: "two"}
	want := []slackapitest.Call{post(1, "Thinking…"), update(1, "one"), failed, post(2, "three")}
	if got := f.Snapshot(); !slices.Equal(got, want) {
		t.Errorf("calls =\n%+v\nwant\n%+v", got, want)
	}
	if d.delivered != 2 {
		t.Errorf("delivered = %d, want 2", d.delivered)
	}
}

func TestDelivererLogsFailedPostWithoutText(t *testing.T) {
	f := slackapitest.New(t)
	f.Fail = map[string][]slackapitest.Response{"chat.postMessage": {{Status: http.StatusInternalServerError}}}
	var buf bytes.Buffer
	d := newTestDeliverer(f, "")
	d.logger = slog.New(slog.NewTextHandler(&buf, nil))
	d.text(t.Context(), "private answer")
	d.finish(t.Context(), nil)

	records := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(records) != 1 {
		t.Fatalf("got %d log records, want 1:\n%s", len(records), buf.String())
	}
	for _, want := range []string{"level=WARN", "method=chat.postMessage", "channel=" + testChannel, "thread_ts=" + testThread} {
		if !strings.Contains(records[0], want) {
			t.Errorf("log record %q does not contain %q", records[0], want)
		}
	}
	if strings.Contains(buf.String(), "private answer") {
		t.Error("log contains the message text")
	}
}

func TestDelivererLogsFailedUpdateAndDelete(t *testing.T) {
	f := slackapitest.New(t)
	f.Fail = map[string][]slackapitest.Response{
		"chat.update": {{Status: http.StatusInternalServerError}},
		"chat.delete": {{APIError: "message_not_found"}},
	}
	var buf bytes.Buffer
	d := newTestDeliverer(f, "Thinking…")
	d.logger = slog.New(slog.NewTextHandler(&buf, nil))
	d.start(t.Context())
	d.text(t.Context(), "one")
	d.finish(t.Context(), nil)

	records := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(records) != 2 {
		t.Fatalf("got %d log records, want 2:\n%s", len(records), buf.String())
	}
	for i, method := range []string{"chat.update", "chat.delete"} {
		if !strings.Contains(records[i], "level=WARN") || !strings.Contains(records[i], "method="+method) {
			t.Errorf("record %d = %q, want a WARN for %s", i, records[i], method)
		}
	}
}

func TestDelivererChunksLongParts(t *testing.T) {
	// "é" is two bytes, so a byte-based limit would split these earlier.
	paragraph := strings.Repeat("é", 3990) + "\n\n" + strings.Repeat("é", 9)
	noBreaks := strings.Repeat("é", 8100)
	tests := []struct {
		name string
		part string
		want []string
	}{
		{
			name: "exactly the limit is one message",
			part: strings.Repeat("é", 4000),
			want: []string{strings.Repeat("é", 4000)},
		},
		{
			name: "exactly the limit with a paragraph break is one message",
			part: strings.Repeat("é", 3990) + "\n\n" + strings.Repeat("é", 8),
			want: []string{strings.Repeat("é", 3990) + "\n\n" + strings.Repeat("é", 8)},
		},
		{
			name: "one over splits at the paragraph break",
			part: paragraph,
			want: []string{strings.Repeat("é", 3990), strings.Repeat("é", 9)},
		},
		{
			name: "line break when there is no paragraph break",
			part: strings.Repeat("é", 3995) + "\n" + strings.Repeat("é", 10),
			want: []string{strings.Repeat("é", 3995), strings.Repeat("é", 10)},
		},
		{
			name: "paragraph break wins over a later line break",
			part: strings.Repeat("é", 3000) + "\n\n" + strings.Repeat("é", 990) + "\n" + strings.Repeat("é", 100),
			want: []string{strings.Repeat("é", 3000), strings.Repeat("é", 990) + "\n" + strings.Repeat("é", 100)},
		},
		{
			name: "a break at the start makes no empty message",
			part: "\n\n" + strings.Repeat("é", 4001),
			want: []string{strings.Repeat("é", 4000), strings.Repeat("é", 1)},
		},
		{
			name: "a lone line break at the start makes no empty message",
			part: "\n" + strings.Repeat("é", 4001),
			want: []string{strings.Repeat("é", 4000), strings.Repeat("é", 1)},
		},
		{
			name: "the last of two paragraph breaks is the cut",
			part: strings.Repeat("é", 1000) + "\n\n" + strings.Repeat("é", 2000) + "\n\n" + strings.Repeat("é", 2000),
			want: []string{strings.Repeat("é", 1000) + "\n\n" + strings.Repeat("é", 2000), strings.Repeat("é", 2000)},
		},
		{
			name: "a trailing break past the limit makes no blank message",
			part: strings.Repeat("é", 4000) + "\n\n",
			want: []string{strings.Repeat("é", 4000)},
		},
		{
			name: "no breaks cuts at the limit",
			part: noBreaks,
			want: []string{noBreaks[:2*4000], noBreaks[2*4000 : 2*8000], noBreaks[2*8000:]},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := slackapitest.New(t)
			d := newTestDeliverer(f, "")
			d.text(t.Context(), tt.part)

			var want []slackapitest.Call
			for i, s := range tt.want {
				want = append(want, post(i+1, s))
			}
			got := f.Snapshot()
			if !slices.Equal(got, want) {
				lens := make([]int, len(got))
				for i, c := range got {
					lens[i] = utf8.RuneCountInString(c.Text)
				}
				t.Errorf("got %d messages of %v runes, want %d", len(got), lens, len(want))
			}
			if d.delivered != 1 {
				t.Errorf("delivered = %d, want 1", d.delivered)
			}
		})
	}
}

// Slack rejects a message with no visible text.
func TestDelivererSkipsBlankText(t *testing.T) {
	for _, part := range []string{"", " \n\n "} {
		f := slackapitest.New(t)
		d := newTestDeliverer(f, "")
		d.text(t.Context(), part)
		if got := f.Snapshot(); len(got) != 0 {
			t.Errorf("text(%q) made calls %+v, want none", part, got)
		}
		if d.delivered != 0 {
			t.Errorf("text(%q): delivered = %d, want 0", part, d.delivered)
		}
	}
}

func TestDelivererRendersPartsOnly(t *testing.T) {
	f := slackapitest.New(t)
	d := newTestDeliverer(f, "**wait**")
	d.errorText = "**failed**"
	d.render = mrkdwn.Render
	d.start(t.Context())
	d.text(t.Context(), "**x**")
	d.finish(t.Context(), errors.New("boom"))

	want := []slackapitest.Call{post(1, "**wait**"), update(1, "*x*"), post(2, "**failed**")}
	if got := f.Snapshot(); !slices.Equal(got, want) {
		t.Errorf("calls =\n%+v\nwant\n%+v", got, want)
	}
}
