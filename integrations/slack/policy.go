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

import "context"

// Decision says whether the service answers a message, and why not.
type Decision struct {
	Allow  bool
	Reason string // logged when !Allow. A Policy's empty Reason is logged as policy_refused.
}

// Policy is the user hook, consulted after the built-in rules.
type Policy func(ctx context.Context, m *Message) (Decision, error)

// Reasons a message is not answered.
const (
	ReasonKillSwitch    = "kill_switch"
	ReasonChannel       = "channel_not_allowed"
	ReasonNoMention     = "no_mention"
	ReasonPolicyError   = "policy_error"
	ReasonPolicyRefused = "policy_refused" // Policy refused without a reason
	ReasonDuplicate     = "duplicate_event"
	ReasonFromBot       = "from_bot"
	ReasonMentionEcho   = "mention_echo" // channel message that is also delivered as app_mention
	ReasonQueueFull     = "queue_full"
	ReasonShutdown      = "shutdown" // accepted before Close, not yet started
)

type gateConfig struct {
	disabled                bool
	channels                map[string]bool // empty = all
	allowUnmentioned        bool
	requireMentionInThreads bool
	// botUserID is a func because the ID is learned lazily, from the first
	// envelope's authorizations or from auth.test.
	botUserID func() string
}

// builtinGate applies the built-in rules in order: kill switch, channel
// allowlist, mention requirement.
func builtinGate(cfg gateConfig, m *Message) Decision {
	if cfg.disabled {
		return Decision{Reason: ReasonKillSwitch}
	}
	if len(cfg.channels) > 0 && !cfg.channels[m.Channel] {
		return Decision{Reason: ReasonChannel}
	}
	if !m.IsMention && !mentionExempt(cfg, m) {
		return Decision{Reason: ReasonNoMention}
	}
	return Decision{Allow: true}
}

// mentionExempt reports whether m may be answered without a mention. A reply
// in a thread whose root message the bot posted continues that conversation.
// The bot's replies go inside the user's thread, so a thread a user started
// is not exempt.
func mentionExempt(cfg gateConfig, m *Message) bool {
	if m.ChannelType == "im" || cfg.allowUnmentioned {
		return true
	}
	return !cfg.requireMentionInThreads && m.ThreadTS != "" && m.ParentUser != "" &&
		m.ParentUser == cfg.botUserID()
}
