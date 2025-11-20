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

// Package telemetry allows to set up custom telemetry processors that the ADK events
// will be emitted to.
package telemetry

import (
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	internaltelemetry "google.golang.org/adk/internal/telemetry"
)

// RegisterSpanProcessor registers the span processor to local trace provider instance.
// Any processor should be registered BEFORE any of the events are emitted, otherwise
// the registration will be ignored.
// In addition to the RegisterSpanProcessor function, global trace provider configs
// are respected.
func RegisterSpanProcessor(processor sdktrace.SpanProcessor) {
	internaltelemetry.AddSpanProcessor(processor)
}
