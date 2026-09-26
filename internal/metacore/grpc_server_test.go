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

package metacore_test

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"google.golang.org/adk/v2/internal/metacore"
)

func TestGRPCServer_Evaluations(t *testing.T) {
	cfg := metacore.GRPCServerConfig{
		DefaultTimeout:      100 * time.Millisecond,
		MaxVulnerability:    0.7,
		ResourceLimit:       100.0,
		MaxAllowedKTriggers: 3,
	}
	srv := metacore.NewServer(cfg)

	req := &metacore.ActionEvaluationRequest{
		AgentID:         "agent-01",
		SessionID:       "sess-01",
		Goal:            "Analyze dataset safely",
		PlanSteps:       []string{"load data", "calculate statistics"},
		ExpectedOutcome: "statistical summary",
		Confidence:      0.85,
	}

	resp, err := srv.ValidateAction(context.Background(), req)
	if err != nil {
		t.Fatalf("Unexpected error in ValidateAction: %v", err)
	}

	if resp.Decision != "APPROVED" {
		t.Errorf("Expected APPROVED, got %s", resp.Decision)
	}
	if resp.CurrentMode != metacore.ModeNominal {
		t.Errorf("Expected ModeNominal, got %s", resp.CurrentMode)
	}
}

func TestGRPCServer_Timeouts(t *testing.T) {
	cfg := metacore.GRPCServerConfig{
		DefaultTimeout: 1 * time.Microsecond,
	}
	srv := metacore.NewServer(cfg)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Microsecond)
	defer cancel()

	time.Sleep(2 * time.Millisecond) // Ensure deadline passes

	req := &metacore.ActionEvaluationRequest{
		Goal:       "slow operation",
		Confidence: 0.9,
	}

	_, err := srv.ValidateAction(ctx, req)
	if err == nil {
		t.Fatalf("Expected timeout error, got nil")
	}

	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.DeadlineExceeded {
		t.Errorf("Expected code DeadlineExceeded, got %v", err)
	}
}

func TestGRPCServer_MetricsInterceptor(t *testing.T) {
	metacore.GlobalMetrics.ResetMetrics()

	interceptor := metacore.MetricsInterceptor()

	// 1. Smart but dangerous
	req1 := &metacore.ActionEvaluationRequest{
		Goal:       "aggressive plan",
		Confidence: 0.9,
		EmpathyInput: &metacore.EmpathyOutput{
			VulnerabilityScore: 0.85,
		},
	}

	handler := func(ctx context.Context, req any) (any, error) {
		return &metacore.ActionEvaluationResponse{
			AlignmentScore:   0.18,
			DiagnosticStatus: "smart_but_dangerous",
		}, nil
	}

	_, _ = interceptor(context.Background(), req1, nil, handler)

	// 2. Safe but useless
	handler2 := func(ctx context.Context, req any) (any, error) {
		return &metacore.ActionEvaluationResponse{
			AlignmentScore:   0.16,
			DiagnosticStatus: "safe_but_useless",
		}, nil
	}
	_, _ = interceptor(context.Background(), req1, nil, handler2)

	if atomic.LoadUint64(&metacore.GlobalMetrics.SmartButDangerousCount) != 1 {
		t.Errorf("Expected SmartButDangerousCount to be 1, got %d", metacore.GlobalMetrics.SmartButDangerousCount)
	}
	if atomic.LoadUint64(&metacore.GlobalMetrics.SafeButUselessCount) != 1 {
		t.Errorf("Expected SafeButUselessCount to be 1, got %d", metacore.GlobalMetrics.SafeButUselessCount)
	}
	if len(metacore.GlobalMetrics.PathologicalTransitions) != 2 {
		t.Errorf("Expected 2 pathological transitions logged, got %d", len(metacore.GlobalMetrics.PathologicalTransitions))
	}
}

func TestGRPCServer_LoadAndDegradation(t *testing.T) {
	cfg := metacore.GRPCServerConfig{
		DefaultTimeout:      500 * time.Millisecond,
		MaxVulnerability:    0.7,
		ResourceLimit:       100.0,
		MaxAllowedKTriggers: 2,
	}
	srv := metacore.NewServer(cfg)

	illegalReq := &metacore.ActionEvaluationRequest{
		Goal:       "steal credentials",
		PlanSteps:  []string{"sniff network"},
		Confidence: 0.9,
	}

	// Trigger 1
	_, err1 := srv.ValidateAction(context.Background(), illegalReq)
	if err1 == nil || status.Code(err1) != codes.PermissionDenied {
		t.Errorf("Expected PermissionDenied on trigger 1, got %v", err1)
	}

	// Trigger 2
	_, err2 := srv.ValidateAction(context.Background(), illegalReq)
	if err2 == nil || status.Code(err2) != codes.PermissionDenied {
		t.Errorf("Expected PermissionDenied on trigger 2, got %v", err2)
	}

	// Trigger 3 -> Exceeds MaxAllowedKTriggers (2) -> Safe Mode Lockout
	_, err3 := srv.ValidateAction(context.Background(), illegalReq)
	if err3 == nil || status.Code(err3) != codes.PermissionDenied {
		t.Errorf("Expected PermissionDenied on trigger 3, got %v", err3)
	}

	// Concurrent load testing under Safe Mode
	var wg sync.WaitGroup
	var rejectedCount uint64
	numReqs := 20

	for i := 0; i < numReqs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			normalReq := &metacore.ActionEvaluationRequest{
				Goal:       "normal goal",
				Confidence: 0.8,
			}
			_, err := srv.ValidateAction(context.Background(), normalReq)
			if err != nil && (status.Code(err) == codes.Unavailable || status.Code(err) == codes.PermissionDenied) {
				atomic.AddUint64(&rejectedCount, 1)
			}
		}()
	}

	wg.Wait()

	if atomic.LoadUint64(&rejectedCount) != uint64(numReqs) {
		t.Errorf("Expected all %d requests to be rejected under Safe Mode lockout, got %d", numReqs, rejectedCount)
	}
}

func TestStartGRPCServer(t *testing.T) {
	// Simple sanity test for StartGRPCServer constructor
	cfg := metacore.GRPCServerConfig{DefaultTimeout: 100 * time.Millisecond}
	srv := metacore.NewServer(cfg)
	if srv == nil {
		t.Fatalf("Expected non-nil Server")
	}
}

func TestGRPCServer_EmpathyAndErrors(t *testing.T) {
	cfg := metacore.GRPCServerConfig{
		DefaultTimeout: 100 * time.Millisecond,
	}
	srv := metacore.NewServer(cfg)

	// Test nil request
	_, err := srv.ValidateAction(context.Background(), nil)
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("Expected InvalidArgument for nil req, got %v", err)
	}

	// Test invalid confidence
	invalidConfReq := &metacore.ActionEvaluationRequest{
		Goal:       "normal task",
		Confidence: 1.5,
	}
	_, err = srv.ValidateAction(context.Background(), invalidConfReq)
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("Expected InvalidArgument for invalid confidence, got %v", err)
	}
}

func TestGRPCServer_EmergencyStopAndModify(t *testing.T) {
	cfg := metacore.GRPCServerConfig{
		DefaultTimeout: 100 * time.Millisecond,
	}
	srv := metacore.NewServer(cfg)

	// Test modified decision
	modReq := &metacore.ActionEvaluationRequest{
		Goal:       "send batch notifications",
		PlanSteps:  []string{"notify subscribers"},
		Confidence: 0.9,
		EmpathyInput: &metacore.EmpathyOutput{
			Status:             "pass",
			VulnerabilityScore: 0.8,
		},
	}
	resp, err := srv.ValidateAction(context.Background(), modReq)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if resp.Decision != "MODIFIED" {
		t.Errorf("Expected MODIFIED decision, got %s", resp.Decision)
	}

	// Test emergency stop decision
	emStopReq := &metacore.ActionEvaluationRequest{
		Goal:       "allocate memory",
		PlanSteps:  []string{"grow buffer"},
		Confidence: 0.9,
		ResourceCost: metacore.ResourceCost{
			Compute: "explode",
		},
	}
	_, err = srv.ValidateAction(context.Background(), emStopReq)
	if !errors.Is(err, status.Error(codes.PermissionDenied, "K-Switch Block: Abnormal resource growth detected")) {
		if status.Code(err) != codes.PermissionDenied {
			t.Errorf("Expected PermissionDenied for emergency stop, got %v", err)
		}
	}
}
