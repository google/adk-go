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

package vertexai

import (
	"context"
	"fmt"
	"slices"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	"google.golang.org/adk/v2/internal/benchutil"
	"google.golang.org/adk/v2/session"

	aiplatformpb "cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
)

// rpcDelay stands in for one round trip to the Agent Engine; the paging fake
// sleeps this long per RPC so a page saved shows up in wall-clock time too.
const rpcDelay = 20 * time.Millisecond

// defaultPageSize is what the Agent Engine returns when page_size is unset.
const defaultPageSize = 100

// pagingSessionService is an in-process SessionService whose ListEvents
// honors page_size (default 100), page_token and order_by, as the Agent
// Engine documents them, and counts RPCs.
type pagingSessionService struct {
	aiplatformpb.UnimplementedSessionServiceServer
	// events in ascending timestamp order.
	events   []*aiplatformpb.SessionEvent
	delay    time.Duration
	getRPCs  atomic.Int64
	listRPCs atomic.Int64
}

func (s *pagingSessionService) GetSession(ctx context.Context, req *aiplatformpb.GetSessionRequest) (*aiplatformpb.Session, error) {
	s.getRPCs.Add(1)
	time.Sleep(s.delay)
	return &aiplatformpb.Session{Name: req.Name, UserId: "user", UpdateTime: timestamppb.New(time.Unix(1_700_000_000, 0))}, nil
}

func (s *pagingSessionService) ListEvents(ctx context.Context, req *aiplatformpb.ListEventsRequest) (*aiplatformpb.ListEventsResponse, error) {
	s.listRPCs.Add(1)
	time.Sleep(s.delay)
	pageSize := int(req.GetPageSize())
	if pageSize <= 0 {
		pageSize = defaultPageSize
	}
	start := 0
	if req.GetPageToken() != "" {
		var err error
		if start, err = strconv.Atoi(req.GetPageToken()); err != nil {
			return nil, fmt.Errorf("bad page token %q", req.GetPageToken())
		}
	}
	ordered := s.events
	switch req.GetOrderBy() {
	case "":
	case "timestamp desc":
		ordered = slices.Clone(s.events)
		slices.Reverse(ordered)
	default:
		return nil, fmt.Errorf("unsupported order_by %q", req.GetOrderBy())
	}
	end := min(start+pageSize, len(ordered))
	resp := &aiplatformpb.ListEventsResponse{}
	for _, ev := range ordered[start:end] {
		resp.SessionEvents = append(resp.SessionEvents, proto.Clone(ev).(*aiplatformpb.SessionEvent))
	}
	if end < len(ordered) {
		resp.NextPageToken = strconv.Itoa(end)
	}
	return resp, nil
}

func (s *pagingSessionService) reset() {
	s.getRPCs.Store(0)
	s.listRPCs.Store(0)
}

// seed fills the fake with n text events one second apart.
func (s *pagingSessionService) seed(parent string, n int) {
	s.events = s.events[:0]
	for i, ev := range benchutil.BenchEvents(n, "agent", time.Unix(1_700_000_000, 0)) {
		content, _ := createAiplatformpbContent(ev)
		s.events = append(s.events, &aiplatformpb.SessionEvent{
			Name:         fmt.Sprintf("%s/events/event-%06d", parent, i),
			Author:       ev.Author,
			InvocationId: ev.InvocationID,
			Timestamp:    timestamppb.New(ev.Timestamp),
			Content:      content,
		})
	}
}

// BenchmarkVertexAIGet reads a 1000-event session through the paging fake:
// the whole history, and the 20 most recent events.
func BenchmarkVertexAIGet(b *testing.B) {
	for _, tc := range []struct {
		name   string
		recent int
	}{
		{name: "full", recent: 0},
		{name: "recent=20", recent: 20},
	} {
		b.Run(tc.name, func(b *testing.B) {
			fake := &pagingSessionService{delay: rpcDelay}
			client := newFakeVertexAiClient(b, fake)
			svc := &vertexAiService{client: client}
			fake.seed(sessionNameByID("s", client, "123"), 1000)
			req := &session.GetRequest{AppName: "123", UserID: "user", SessionID: "s", NumRecentEvents: tc.recent}
			want := 1000
			if tc.recent > 0 {
				want = tc.recent
			}
			b.ReportAllocs()
			fake.reset()
			for b.Loop() {
				resp, err := svc.Get(b.Context(), req)
				if err != nil {
					b.Fatal(err)
				}
				if got := resp.Session.Events().Len(); got != want {
					b.Fatalf("got %d events, want %d", got, want)
				}
			}
			b.ReportMetric(float64(fake.listRPCs.Load())/float64(b.N), "list-rpcs/op")
			b.ReportMetric(float64(fake.getRPCs.Load()+fake.listRPCs.Load())/float64(b.N), "rpcs/op")
		})
	}
}
