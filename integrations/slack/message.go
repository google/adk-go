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
	"encoding/json"
	"fmt"
	"strings"
)

// Message is one Slack event normalized for the service, independent of the
// transport it arrived on.
type Message struct {
	EventID     string
	TeamID      string
	Channel     string
	ChannelType string // "im", "channel", "group", "mpim"
	UserID      string // author
	Text        string // raw Slack text, mention tokens included
	TS          string
	ThreadTS    string // empty for a thread root
	IsMention   bool   // app_mention, or a message whose text mentions the bot
	FromBot     bool   // bot_id or bot_profile set
	ParentUser  string // parent_user_id when present
	Retry       int    // X-Slack-Retry-Num, 0 when absent

	// kind is the Slack event type. Slack delivers a mention both as
	// app_mention and as a plain message, and this tells the two apart.
	kind string
}

// threadKey returns the thread key: ThreadTS, or TS for a root message.
func (m *Message) threadKey() string {
	if m.ThreadTS != "" {
		return m.ThreadTS
	}
	return m.TS
}

// SessionID returns the ADK session id for a Slack thread. It matches
// ^[A-Za-z0-9_-]+$, because Vertex managed sessions reject any other
// character (see https://github.com/google/adk-python/issues/6435), so the
// dot in a Slack timestamp becomes an underscore.
func SessionID(channel, threadTS string) string {
	var b strings.Builder
	for _, r := range channel + "-" + threadTS {
		switch {
		case r >= 'A' && r <= 'Z', r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return b.String()
}

// envelope is the Events API body (also the payload of a Socket Mode
// events_api envelope).
type envelope struct {
	Type           string          `json:"type"`
	Challenge      string          `json:"challenge"`
	EventID        string          `json:"event_id"`
	TeamID         string          `json:"team_id"`
	Authorizations []authorization `json:"authorizations"`
	Event          json.RawMessage `json:"event"`
}

type authorization struct {
	UserID string `json:"user_id"`
	IsBot  bool   `json:"is_bot"`
}

// parseEventCallback turns an event_callback envelope into a Message.
// botUserID is used to detect mentions inside plain messages. It may be empty.
// Returns (nil, nil) for event types the service ignores.
func parseEventCallback(env *envelope, botUserID string) (*Message, error) {
	var ev struct {
		Type         string `json:"type"`
		Subtype      string `json:"subtype"`
		User         string `json:"user"`
		Text         string `json:"text"`
		Channel      string `json:"channel"`
		ChannelType  string `json:"channel_type"`
		TS           string `json:"ts"`
		ThreadTS     string `json:"thread_ts"`
		BotID        string `json:"bot_id"`
		BotProfile   any    `json:"bot_profile"`
		ParentUserID string `json:"parent_user_id"`
	}
	if err := json.Unmarshal(env.Event, &ev); err != nil {
		return nil, fmt.Errorf("slack: decode event: %w", err)
	}
	var mention bool
	switch ev.Type {
	case "app_mention":
		mention = true
	case "message":
		mention = botUserID != "" && strings.Contains(ev.Text, "<@"+botUserID+">")
	default:
		return nil, nil
	}
	// A broadcast is a real reply that also shows in the channel, and a file
	// share with text is a message with an attachment. Other subtypes, such
	// as edits and joins, are not prompts.
	switch ev.Subtype {
	case "", "thread_broadcast":
	case "file_share":
		if ev.Text == "" {
			return nil, nil
		}
	default:
		return nil, nil
	}
	return &Message{
		EventID:     env.EventID,
		TeamID:      env.TeamID,
		Channel:     ev.Channel,
		ChannelType: channelType(ev.ChannelType, ev.Channel),
		UserID:      ev.User,
		Text:        ev.Text,
		TS:          ev.TS,
		ThreadTS:    ev.ThreadTS,
		IsMention:   mention,
		FromBot:     ev.BotID != "" || ev.BotProfile != nil,
		ParentUser:  ev.ParentUserID,
		kind:        ev.Type,
	}, nil
}

// channelType returns typ, or infers it from the channel ID's prefix when
// Slack omits it, as it does for app_mention.
func channelType(typ, channel string) string {
	if typ != "" {
		return typ
	}
	switch {
	case strings.HasPrefix(channel, "D"):
		return "im"
	case strings.HasPrefix(channel, "C"):
		return "channel"
	case strings.HasPrefix(channel, "G"):
		return "group"
	}
	return ""
}
