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

package memory_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"google.golang.org/adk/v2/memory"
)

// recordingMemoryAdder implements memory.MemoryAdder. It records every request
// it receives and returns err.
type recordingMemoryAdder struct {
	eventlessService
	err  error
	reqs []*memory.AddMemoryRequest
}

func (r *recordingMemoryAdder) AddMemory(_ context.Context, req *memory.AddMemoryRequest) error {
	r.reqs = append(r.reqs, req)
	return r.err
}

func TestAddMemoryDelegatesToMemoryAdder(t *testing.T) {
	errBoom := errors.New("boom")
	for _, tt := range []struct {
		name     string
		adderErr error
		wantErr  error
	}{
		{name: "adder succeeds", adderErr: nil, wantErr: nil},
		{name: "adder error is returned", adderErr: errBoom, wantErr: errBoom},
	} {
		t.Run(tt.name, func(t *testing.T) {
			adder := &recordingMemoryAdder{err: tt.adderErr}
			req := &memory.AddMemoryRequest{
				AppName:        "app-1",
				UserID:         "user-1",
				Memories:       []*memory.Entry{{ID: "m1"}},
				CustomMetadata: map[string]any{"k": "v"},
			}

			err := memory.AddMemory(t.Context(), adder, req)

			if !errors.Is(err, tt.wantErr) {
				t.Errorf("AddMemory() error = %v, want %v", err, tt.wantErr)
			}
			if len(adder.reqs) != 1 {
				t.Fatalf("MemoryAdder.AddMemory called %d times, want 1", len(adder.reqs))
			}
			if adder.reqs[0] != req {
				t.Errorf("MemoryAdder.AddMemory got request %p, want the caller's request %p", adder.reqs[0], req)
			}
		})
	}
}

func TestAddMemoryUnsupportedService(t *testing.T) {
	for _, tt := range []struct {
		name string
		svc  memory.Service
	}{
		{name: "service without MemoryAdder", svc: eventlessService{}},
		{name: "in-memory service", svc: memory.InMemoryService()},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := memory.AddMemory(t.Context(), tt.svc, &memory.AddMemoryRequest{
				AppName:  "app-1",
				UserID:   "user-1",
				Memories: []*memory.Entry{{ID: "m1"}},
			})

			if !errors.Is(err, memory.ErrAddMemoryUnsupported) {
				t.Errorf("AddMemory() error = %v, want errors.Is(err, memory.ErrAddMemoryUnsupported)", err)
			}
			if !errors.Is(err, errors.ErrUnsupported) {
				t.Errorf("AddMemory() error = %v, want errors.Is(err, errors.ErrUnsupported)", err)
			}
			// Service has no AddEventsToMemory method; adding events goes
			// through the package function memory.AddEventsToMemory.
			if err != nil && strings.Contains(err.Error(), "Service.AddEventsToMemory") {
				t.Errorf("AddMemory() error = %q, want it not to name the nonexistent %q", err, "Service.AddEventsToMemory")
			}
		})
	}
}

func TestAddMemoryNilArguments(t *testing.T) {
	adder := &recordingMemoryAdder{}
	for _, tt := range []struct {
		name    string
		svc     memory.Service
		req     *memory.AddMemoryRequest
		wantErr string
	}{
		{name: "nil service", svc: nil, req: &memory.AddMemoryRequest{}, wantErr: "memory: service must not be nil"},
		{name: "nil request", svc: adder, req: nil, wantErr: "memory: AddMemoryRequest must not be nil"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := memory.AddMemory(t.Context(), tt.svc, tt.req)

			if err == nil || err.Error() != tt.wantErr {
				t.Errorf("AddMemory() error = %v, want %q", err, tt.wantErr)
			}
			if errors.Is(err, memory.ErrAddMemoryUnsupported) {
				t.Errorf("AddMemory() error = %v, want an error other than ErrAddMemoryUnsupported", err)
			}
		})
	}
	if len(adder.reqs) != 0 {
		t.Errorf("MemoryAdder.AddMemory called %d times, want 0", len(adder.reqs))
	}
}
