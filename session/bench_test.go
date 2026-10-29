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

package session_test

import (
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/v2/internal/benchutil"
	"google.golang.org/adk/v2/session"
)

var benchSizes = []int{10, 100, 1000, 10000}

func BenchmarkInMemoryGet(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			svc := session.InMemoryService()
			if _, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", n); err != nil {
				b.Fatal(err)
			}
			req := &session.GetRequest{AppName: "app", UserID: "user", SessionID: "s"}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := svc.Get(b.Context(), req); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkInMemoryAppendEvent appends to a session that already holds n
// events, which is what every turn of a long conversation does.
func BenchmarkInMemoryAppendEvent(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("events=%d", n), func(b *testing.B) {
			svc := session.InMemoryService()
			sess, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "agent", n)
			if err != nil {
				b.Fatal(err)
			}
			evs := benchutil.BenchEvents(2, "agent", time.Unix(1_800_000_000, 0))
			b.ReportAllocs()
			b.ResetTimer()
			i := 0
			for b.Loop() {
				ev := *evs[i%2]
				ev.ID = ""
				if err := svc.AppendEvent(b.Context(), sess, &ev); err != nil {
					b.Fatal(err)
				}
				i++
			}
		})
	}
}

func BenchmarkInMemoryList(b *testing.B) {
	for _, sessions := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("sessions=%d", sessions), func(b *testing.B) {
			svc := session.InMemoryService()
			for i := range sessions {
				if _, err := benchutil.SeedSession(b.Context(), svc, "app", "user", fmt.Sprintf("s%d", i), "agent", 20); err != nil {
					b.Fatal(err)
				}
			}
			req := &session.ListRequest{AppName: "app", UserID: "user"}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := svc.List(b.Context(), req); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkInMemoryAppendEventParallel appends from GOMAXPROCS goroutines at
// once, each to its own session of one service holding `sessions` sessions,
// the way concurrent turns of different users do. With no shared bottleneck
// ns/op is BenchmarkInMemoryAppendEvent divided by the core count.
func BenchmarkInMemoryAppendEventParallel(b *testing.B) {
	for _, sessions := range []int{100, 10000} {
		b.Run(fmt.Sprintf("sessions=%d", sessions), func(b *testing.B) {
			svc := session.InMemoryService()
			created := make([]session.Session, sessions)
			for i := range created {
				resp, err := svc.Create(b.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: fmt.Sprintf("s%d", i)})
				if err != nil {
					b.Fatal(err)
				}
				created[i] = resp.Session
			}
			evs := benchutil.BenchEvents(2, "agent", time.Unix(1_800_000_000, 0))
			var next atomic.Int64
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				// One session per goroutine: a session's own appends are
				// sequential, so only the service is shared.
				sess := created[int(next.Add(1)-1)%sessions]
				i := 0
				for pb.Next() {
					ev := *evs[i%2]
					ev.ID = ""
					if err := svc.AppendEvent(b.Context(), sess, &ev); err != nil {
						b.Error(err)
						return
					}
					i++
				}
			})
		})
	}
}
