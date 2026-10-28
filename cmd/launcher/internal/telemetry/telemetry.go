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

// Package telemetry contains the internal shared logic for initializing telemetry in launchers.
package telemetry

import (
	"context"
	"slices"

	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/telemetry"
)

// InitAndSetGlobalOtelProviders initializes telemetry and sets the global OTel providers.
//
// otelToCloud is the launcher's -otel_to_cloud flag. Set, it turns export to
// GCP on. Unset, it leaves the choice to config.TelemetryOptions, so the flag's
// default does not override a [telemetry.WithOtelToCloud] the embedder passed.
func InitAndSetGlobalOtelProviders(ctx context.Context, config *launcher.Config, otelToCloud bool) (*telemetry.Providers, error) {
	// A copy: appending to config.TelemetryOptions could write into the
	// embedder's backing array.
	opts := slices.Clone(config.TelemetryOptions)
	if otelToCloud {
		opts = append(opts, telemetry.WithOtelToCloud(true))
	}
	telemetryProviders, err := telemetry.New(ctx, opts...)
	if err != nil {
		return nil, err
	}
	telemetryProviders.SetGlobalOtelProviders()
	return telemetryProviders, nil
}
