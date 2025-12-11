// Copyright 2025 Google LLC
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

package runconfig

import "context"

type StreamingMode string

const (
	StreamingModeNone StreamingMode = "none"
	StreamingModeSSE  StreamingMode = "sse"
	StreamingModeBidi StreamingMode = "bidi"
)

type RunConfig struct {
	StreamingMode StreamingMode
}

func ToContext(ctx context.Context, cfg *RunConfig) context.Context {
	return context.WithValue(ctx, runConfigCtxKey, cfg)
}

func FromContext(ctx context.Context) *RunConfig {
	m, ok := ctx.Value(runConfigCtxKey).(*RunConfig)
	if !ok {
		return nil
	}
	return m
}

type ctxKey int

const runConfigCtxKey ctxKey = 0
