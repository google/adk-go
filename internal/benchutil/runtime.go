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

package benchutil

import (
	"runtime/metrics"
	"testing"
	"time"
)

// RuntimeSample is a snapshot of the process counters a turn benchmark
// reports per iteration next to wall time.
type RuntimeSample struct {
	cpu        time.Duration
	goroutines uint64
}

// SampleRuntime takes a RuntimeSample.
func SampleRuntime() RuntimeSample {
	s := []metrics.Sample{{Name: "/sched/goroutines-created:goroutines"}}
	metrics.Read(s)
	var created uint64
	if s[0].Value.Kind() == metrics.KindUint64 {
		created = s[0].Value.Uint64()
	}
	return RuntimeSample{cpu: CPUTime(), goroutines: created}
}

// ReportRuntime reports CPU time and goroutines created per iteration since
// start. CPU time covers every goroutine and GC, so for a benchmark that runs
// nothing else it is the CPU cost of one iteration.
func ReportRuntime(b *testing.B, start RuntimeSample) {
	end := SampleRuntime()
	if start.cpu != 0 {
		b.ReportMetric(float64(end.cpu-start.cpu)/float64(b.N), "cpu-ns/op")
	}
	b.ReportMetric(float64(end.goroutines-start.goroutines)/float64(b.N), "goroutines/op")
}
