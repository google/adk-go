// Copyright 2025 The A2A Authors
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

package a2a

// GetTaskPushConfigParams defines parameters for fetching a specific push notification configuration for a task.
type GetTaskPushConfigParams struct {
	// TaskID is the unique identifier of the task.
	TaskID TaskID `json:"id" yaml:"id" mapstructure:"id"`

	// ConfigID is an optional ID of the push notification configuration to retrieve.
	ConfigID string `json:"pushNotificationConfigId,omitempty" yaml:"pushNotificationConfigId,omitempty" mapstructure:"pushNotificationConfigId,omitempty"`

	// Metadata is an optional metadata for extensions.
	Metadata map[string]any `json:"metadata,omitempty" yaml:"metadata,omitempty" mapstructure:"metadata,omitempty"`
}

// ListTaskPushConfigParams defines parameters for listing all push notification configurations associated with a task.
type ListTaskPushConfigParams struct {
	// TaskID is the unique identifier of the task.
	TaskID TaskID `json:"id" yaml:"id" mapstructure:"id"`

	// Metadata is an optional metadata for extensions.
	Metadata map[string]any `json:"metadata,omitempty" yaml:"metadata,omitempty" mapstructure:"metadata,omitempty"`
}

// DeleteTaskPushConfigParams defines parameters for deleting a specific push notification configuration for a task.
type DeleteTaskPushConfigParams struct {
	// TaskID is the unique identifier of the task.
	TaskID TaskID `json:"id" yaml:"id" mapstructure:"id"`

	// ConfigID is the ID of the push notification configuration to delete.
	ConfigID string `json:"pushNotificationConfigId" yaml:"pushNotificationConfigId" mapstructure:"pushNotificationConfigId"`

	// Metadata is an optional metadata for extensions.
	Metadata map[string]any `json:"metadata,omitempty" yaml:"metadata,omitempty" mapstructure:"metadata,omitempty"`
}

// TaskPushConfig is a container associating a push notification configuration with a specific task.
type TaskPushConfig struct {
	// Config is the push notification configuration for this task.
	Config PushConfig `json:"pushNotificationConfig" yaml:"pushNotificationConfig" mapstructure:"pushNotificationConfig"`

	// TaskID is the ID of the task.
	TaskID TaskID `json:"taskId" yaml:"taskId" mapstructure:"taskId"`
}

// PushConfig defines the configuration for setting up push notifications for task updates.
type PushConfig struct {
	// ID is an optional unique ID for the push notification configuration, set by the client
	// to support multiple notification callbacks.
	ID string `json:"id,omitempty" yaml:"id,omitempty" mapstructure:"id,omitempty"`

	// Auth is an optional authentication details for the agent to use when calling the
	// notification URL.
	Auth *PushAuthInfo `json:"authentication,omitempty" yaml:"authentication,omitempty" mapstructure:"authentication,omitempty"`

	// Token is an optional unique token for this task or session to validate incoming push notifications.
	Token string `json:"token,omitempty" yaml:"token,omitempty" mapstructure:"token,omitempty"`

	// URL is the callback URL where the agent should send push notifications.
	URL string `json:"url" yaml:"url" mapstructure:"url"`
}

// PushAuthInfo defines authentication details for a push notification endpoint.
type PushAuthInfo struct {
	// Credentials is an optional credentials required by the push notification endpoint.
	Credentials string `json:"credentials,omitempty" yaml:"credentials,omitempty" mapstructure:"credentials,omitempty"`

	// Schemes is a list of supported authentication schemes (e.g., 'Basic', 'Bearer').
	Schemes []string `json:"schemes" yaml:"schemes" mapstructure:"schemes"`
}
