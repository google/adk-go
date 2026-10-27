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

package telemetry

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/telemetry"
)

// withoutCloudCredentials makes export to GCP fail as soon as it is turned on,
// without network access: the credentials file it names does not exist. The
// error then tells whether export was turned on.
func withoutCloudCredentials(t *testing.T) {
	t.Helper()
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", filepath.Join(t.TempDir(), "missing.json"))
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
	t.Setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "")
}

func initProviders(t *testing.T, config *launcher.Config, otelToCloud bool) (cloudExport bool) {
	t.Helper()
	providers, err := InitAndSetGlobalOtelProviders(t.Context(), config, otelToCloud)
	if err != nil {
		if !strings.Contains(err.Error(), "credentials") {
			t.Fatalf("InitAndSetGlobalOtelProviders() error = %v, want nil or a credentials error", err)
		}
		return true
	}
	t.Cleanup(func() {
		if err := providers.Shutdown(context.Background()); err != nil {
			t.Errorf("Shutdown() error = %v", err)
		}
	})
	return false
}

func TestInitAndSetGlobalOtelProviders_OtelToCloud(t *testing.T) {
	withoutCloudCredentials(t)
	tests := []struct {
		name            string
		options         []telemetry.Option
		flag            bool
		wantCloudExport bool
	}{
		{name: "no flag, no option"},
		{name: "flag", flag: true, wantCloudExport: true},
		{
			name:            "option without flag",
			options:         []telemetry.Option{telemetry.WithOtelToCloud(true)},
			wantCloudExport: true,
		},
		{
			name:            "flag overrides disabling option",
			options:         []telemetry.Option{telemetry.WithOtelToCloud(false)},
			flag:            true,
			wantCloudExport: true,
		},
		{
			name:    "disabling option without flag",
			options: []telemetry.Option{telemetry.WithOtelToCloud(false)},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			config := &launcher.Config{TelemetryOptions: tc.options}
			if got := initProviders(t, config, tc.flag); got != tc.wantCloudExport {
				t.Errorf("export to GCP turned on = %v, want %v", got, tc.wantCloudExport)
			}
		})
	}
}

func TestInitAndSetGlobalOtelProviders_KeepsTelemetryOptions(t *testing.T) {
	withoutCloudCredentials(t)
	// Spare capacity, so an append to the embedder's slice would write into it.
	options := make([]telemetry.Option, 1, 2)
	options[0] = telemetry.WithOtelToCloud(false)

	initProviders(t, &launcher.Config{TelemetryOptions: options}, true)

	if got := options[:cap(options)][1]; got != nil {
		t.Errorf("TelemetryOptions backing array element 1 = %v, want nil: the launcher wrote into the embedder's slice", got)
	}
}
