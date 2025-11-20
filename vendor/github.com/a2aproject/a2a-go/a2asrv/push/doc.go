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

// Package push provides a basic implementation of push notification functionality.
// To enable push notifications in the default server implementation, [github.com/a2aproject/a2a-go/a2asrv.WithPushNotifications] function
// should be used:
//
//	sender := push.NewHTTPPushSender()
//	configStore := push.NewInMemoryStore()
//	requestHandler := a2asrv.NewRequestHandler(
//		agentExecutor,
//		a2asrv.WithPushNotifications(configStore, sender),
//	)
package push
