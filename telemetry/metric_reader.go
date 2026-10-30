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
	"sync/atomic"

	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
)

// newPeriodicReader returns a reader over a [sdkmetric.PeriodicReader] which
// deosn't export on Shutdown, and which exports nothing while no instrument has recorded.
func newPeriodicReader(exporter sdkmetric.Exporter, opts ...sdkmetric.PeriodicReaderOption) sdkmetric.Reader {
	guarded := &guardedExporter{Exporter: exporter}
	return &noFinalExportReader{
		PeriodicReader: sdkmetric.NewPeriodicReader(guarded, opts...),
		exporter:       guarded,
	}
}

// noFinalExportReader wraps [sdkmetric.PeriodicReader] so it doesn't export on Shutdown.
type noFinalExportReader struct {
	*sdkmetric.PeriodicReader
	exporter *guardedExporter
}

func (r *noFinalExportReader) Shutdown(ctx context.Context) error {
	r.exporter.dropExports.Store(true)
	return r.PeriodicReader.Shutdown(ctx)
}

// guardedExporter is a wrapper over [sdkmetric.Exporter] that discards an
// empty collection, and every collection once dropExports is set.
type guardedExporter struct {
	sdkmetric.Exporter
	dropExports atomic.Bool
}

func (e *guardedExporter) Export(ctx context.Context, rm *metricdata.ResourceMetrics) error {
	if e.dropExports.Load() || len(rm.ScopeMetrics) == 0 {
		return nil
	}
	return e.Exporter.Export(ctx, rm)
}
