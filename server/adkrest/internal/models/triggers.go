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

import "encoding/json"

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

// EventarcTriggerRequest represents the request for the Eventarc trigger.
// Eventarc / CloudEvents request format.
//
// Eventarc delivers events as CloudEvents over HTTP in two modes:
//
//  1. **Structured content mode** (JSON body): All CloudEvents attributes
//     and the event data are in the JSON body.  Used by direct HTTP callers.
//  2. **Binary content mode** (Eventarc default): CloudEvents attributes are
//     sent as “ce-*“ HTTP headers, and the body contains only the event
//     data — typically a Pub/Sub message wrapper for Pub/Sub-sourced events:
//     “{"message": {"data": "<base64>", ...}, "subscription": "..."}“.
//
// See: https://cloud.google.com/eventarc/docs/cloudevents
type EventarcTriggerRequest struct {
	// Unique identifier for the event
	ID string `json:"id"`
	// Identifies the source of the event
	Source string `json:"source"`
	// The type of event data
	Type string `json:"type"`
	// The CloudEvents specification version used for this event
	SpecVersion string `json:"specversion"`
	// Event generation time, in RFC 3339 format (optional)
	Time string `json:"time,omitempty"`
	// In structured mode, ``data`` is always present.
	// In binary mode, the entire body is the data (often a Pub/Sub wrapper).
	// json.RawMessage captures the raw JSON payload (Storage, PubSub, whatever)
	// without needing to know its internal structure!
	Data json.RawMessage `json:"data,omitempty"`
}

// TriggerResponse represents the standard response for Pub/Sub and Eventarc triggers.
type TriggerResponse struct {
	// Processing status: 'success' or error message.
	Status string `json:"status"`
}
