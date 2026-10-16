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

package plugin

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
)

func TestNew(t *testing.T) {
	mockOnUserMsg := func(agent.InvocationContext, *genai.Content) (*genai.Content, error) { return nil, nil }
	mockCloseErr := errors.New("close error")

	tests := []struct {
		name           string
		cfg            Config
		validate       func(*testing.T, *Plugin)
		expectCloseErr error
	}{
		{
			name: "Successfully maps all fields",
			cfg: Config{
				OnUserMessageCallback: mockOnUserMsg,
			},
			validate: func(t *testing.T, p *Plugin) {
				if p.OnUserMessageCallback() == nil {
					t.Error("OnUserMessageCallback was not mapped correctly")
				}
				// Verify Close() is safe even if we didn't provide a CloseFunc
				if err := p.Close(); err != nil {
					t.Errorf("Expected nil error from default close, got %v", err)
				}
			},
		},
		{
			name: "Safety: Handles nil CloseFunc gracefully",
			cfg:  Config{
				// No CloseFunc provided
			},
			validate: func(t *testing.T, p *Plugin) {
				// This should not panic
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("Plugin.Close() panicked with nil CloseFunc")
					}
				}()
				if err := p.Close(); err != nil {
					t.Errorf("Expected nil error, got %v", err)
				}
			},
		},
		{
			name: "Functionality: Executes provided CloseFunc",
			cfg: Config{
				CloseFunc: func() error {
					return mockCloseErr
				},
			},
			validate: func(t *testing.T, p *Plugin) {
				err := p.Close()
				if err != mockCloseErr {
					t.Errorf("Expected error '%v', got '%v'", mockCloseErr, err)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := New(tt.cfg)
			if err != nil {
				t.Fatalf("New() returned unexpected error: %v", err)
			}
			if p == nil {
				t.Fatal("New() returned nil Plugin")
			}
			if tt.validate != nil {
				tt.validate(t, p)
			}
		})
	}
}

func TestPluginCloseRunsCloseFuncOnce(t *testing.T) {
	closeErr := errors.New("flush failed")
	var calls atomic.Int32
	p, err := New(Config{
		Name: "counting",
		CloseFunc: func() error {
			calls.Add(1)
			return closeErr
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			if err := p.Close(); !errors.Is(err, closeErr) {
				t.Errorf("Close() error = %v, want %v", err, closeErr)
			}
		})
	}
	wg.Wait()
	if err := p.Close(); !errors.Is(err, closeErr) {
		t.Errorf("Close() after the first calls error = %v, want %v", err, closeErr)
	}

	if got := calls.Load(); got != 1 {
		t.Errorf("CloseFunc ran %d times, want 1", got)
	}
}
