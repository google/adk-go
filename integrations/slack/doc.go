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

// Package slack serves an ADK agent as a Slack app. It receives messages
// through the Slack Events API, runs one agent turn per message, and posts
// the agent's reply into the message's thread.
//
// Each Slack thread is one ADK session (see SessionID), so a conversation in
// a thread keeps its history. Turns of one thread never overlap, and under a
// burst their order is best effort. Turns of different threads run
// concurrently up to MaxConcurrentRuns. A
// burst of messages in one thread can occupy up to MaxConcurrentRuns workers
// while they wait for that thread's turn to finish, because turns are
// serialized per thread after a worker is taken.
//
// # Usage
//
//	backend, err := slack.NewLocalBackend(slack.LocalBackendConfig{
//		AppName: "helper",
//		Agent:   myAgent,
//	})
//	if err != nil {
//		return err
//	}
//	svc, err := slack.New(ctx, slack.Config{
//		BotToken:      os.Getenv("SLACK_BOT_TOKEN"),
//		SigningSecret: os.Getenv("SLACK_SIGNING_SECRET"),
//		Backend:       backend,
//	})
//	if err != nil {
//		return err
//	}
//	defer svc.Close(ctx)
//	http.Handle("/slack/events", svc.Handler())
//
// Set the Slack app's Event Subscriptions Request URL to the handler's
// public URL, and subscribe to the app_mention and message.* bot events.
//
// # Configuration
//
// Apart from the three required fields (BotToken, Backend, and SigningSecret
// for Handler), Config fields fall into three groups:
//
//   - Who is answered: Channels, AllowUnmentioned, RequireMentionInThreads,
//     Disabled and the Policy hook. By default the bot answers direct
//     messages, mentions, and unmentioned replies in threads whose root
//     message the bot posted. The bot replies inside the user's thread, so
//     in a thread a user started, follow-ups must mention the bot again
//     unless AllowUnmentioned is set.
//   - How replies look: NoPlaceholder, Placeholder, ErrorMessage and
//     DisableMarkdownConversion.
//   - How turns run: MaxConcurrentRuns, MaxQueued, TurnTimeout, Deduper,
//     UserIDFunc, HTTPClient and Logger.
//
// # Deployment
//
// The endpoint is public, and Slack's request signature is its only
// authentication. Serve it from a process that does not also expose an
// unauthenticated ADK api or webui server, because anyone who can reach
// those can run the agent and read its sessions.
//
// The default Deduper is in memory, so with several replicas behind a load
// balancer, a retried event can reach a replica that has not seen it. Supply
// a shared Deduper in that case.
package slack
