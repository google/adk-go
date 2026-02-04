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

// Package telemetry implements the open telemetry in ADK.
package telemetry

import (
	"context"
	"os"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/resource"
)

func newProjectIDDetector(cfg *config) resource.Detector {
	return &projectIDDetector{
		cfg: *cfg,
	}
}

var projectIDAttribute = attribute.Key("gcp.project_id")

// projectIDDetector is a resource detector that resolves the gcp.project_id attribute in the following order:
// 1. Resource_project_id from config, if present.
// 2. ProjectID from credentials, if present.
// 3. GOOGLE_CLOUD_PROJECT environment variable, if present.
type projectIDDetector struct {
	cfg config
}

func (d *projectIDDetector) Detect(ctx context.Context) (*resource.Resource, error) {
	projectID := d.resolveProjectID()
	if projectID == "" {
		return nil, nil
	}
	return resource.New(ctx, resource.WithAttributes(projectIDAttribute.String(projectID)))
}

func (d *projectIDDetector) resolveProjectID() string {
	if d.cfg.resourceProjectID != "" {
		return d.cfg.resourceProjectID
	}
	if d.cfg.credentials != nil && d.cfg.credentials.ProjectID != "" {
		return d.cfg.credentials.ProjectID
	}
	return os.Getenv("GOOGLE_CLOUD_PROJECT")
}
