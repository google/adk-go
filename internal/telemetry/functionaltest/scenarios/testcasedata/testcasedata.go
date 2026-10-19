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

// Package testcasedata is the data half of the functional telemetry tests:
// which scenario a case drives, and under which telemetry configuration. Each
// scenario has a package of its own below this one; the testcaseimpl package
// turns a scenario into the agents it runs.
package testcasedata

// Scenario is implemented by the Scenario type in each testcasedata/<name>
// package, which carries the variations only that scenario has, such as its
// own failure modes.
type Scenario interface {
	// Name is the scenario's testdata directory.
	Name() string
	// Variant names what sets this value apart from the scenario's default,
	// or "" for the default. It prefixes the test id.
	Variant() string
	// WantErr reports whether running the scenario yields an error.
	WantErr() bool
}

// Capture is an OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT value.
type Capture string

// The capture modes every scenario is recorded under: adk-python's
// experimental SEMCONV_CONFIGS.
const (
	noContent    Capture = "no_content"
	spanOnly     Capture = "span_only"
	eventOnly    Capture = "event_only"
	spanAndEvent Capture = "span_and_event"
)

// SchemaVersion is an ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN value.
type SchemaVersion string

// The schema versions every scenario is recorded under.
const (
	otelSemconv136 SchemaVersion = "otel_semconv_1_36"
	otelSemconv144 SchemaVersion = "otel_semconv_1_44"
)

// Case is one scenario recorded under one telemetry configuration.
type Case struct {
	Scenario      Scenario
	Capture       Capture
	SchemaVersion SchemaVersion
}

// ID names the case, and its golden: <scenario>/[<variant>-]<capture>-<schema>,
// with the configuration spelled as the environment sets it.
func (c Case) ID() string {
	id := string(c.Capture) + "-" + string(c.SchemaVersion)
	if v := c.Scenario.Variant(); v != "" {
		id = v + "-" + id
	}
	return c.Scenario.Name() + "/" + id
}

// Matrix returns every variant under every capture mode and schema version.
func Matrix(variants ...Scenario) []Case {
	var out []Case
	for _, s := range variants {
		for _, c := range []Capture{noContent, spanOnly, eventOnly, spanAndEvent} {
			for _, v := range []SchemaVersion{otelSemconv136, otelSemconv144} {
				out = append(out, Case{Scenario: s, Capture: c, SchemaVersion: v})
			}
		}
	}
	return out
}
