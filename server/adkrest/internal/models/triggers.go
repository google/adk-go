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

package models

// PubSubTriggerRequest represents the request for the PubSub trigger.
// See: https://cloud.google.com/pubsub/docs/push#receive_push
type PubSubTriggerRequest struct {
	// The Pub/Sub message.
	Message PubSubMessage `json:"message"`
	// The subscription this message was published to.
	Subscription string `json:"subscription"`
}

// PubSubMessage represents the message for the PubSub trigger.
type PubSubMessage struct {
	// The message payload. This will always be a base64-encoded string.
	Data []byte `json:"data"`
	// ID of this message, assigned by the Pub/Sub server.
	MessageID string `json:"messageId"`
	// The time at which the message was published, populated by the server.
	PublishTime string `json:"publishTime"`
	// Optional attributes for this message. An object containing a list of 'key': 'value' string pairs.
	Attributes map[string]string `json:"attributes,omitempty"`
	// If message ordering is enabled, this identifies related messages for which publish order should be respected.
	OrderingKey string `json:"orderingKey,omitempty"`
}

// TriggerResponse represents the standard response for Pub/Sub and Eventarc triggers.
type TriggerResponse struct {
	// Processing status: 'success' or error message.
	Status string `json:"status"`
}
