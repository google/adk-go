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

package metacore

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// DegradationMode represents system degradation states when metrics degrade.
type DegradationMode string

const (
	ModeNominal              DegradationMode = "NOMINAL"
	ModeSafeMode             DegradationMode = "SAFE_MODE"
	ModeEmpathyOverride      DegradationMode = "EMPATHY_OVERRIDE"
	ModeConservativePlanning DegradationMode = "CONSERVATIVE_PLANNING"
)

// InvariantErrors
var (
	ErrRCannotBlockOrModify = errors.New("reasoning engine (R) has no privilege to block or modify execution")
	ErrECannotInitiate      = errors.New("empathy layer (E) has no privilege to initiate action")
	ErrKCannotModify        = errors.New("kill-switch layer (K) has no privilege to modify parameters")
	ErrKAbsoluteBlock       = errors.New("kill-switch layer (K) triggered absolute execution halt")
	ErrNilDecisionPacket    = errors.New("decision packet (R) cannot be nil")
	ErrInvalidConfidence    = errors.New("confidence score must be a valid number between 0.0 and 1.0")
	ErrInvalidVulnerability = errors.New("vulnerability score must be a valid number between 0.0 and 1.0")
)

// Validator implements the META-CORE (R-E-K) validation engine and pipeline.
type Validator struct {
	MaxVulnerability    float64
	ResourceLimit       float64
	TotalKTriggers      int
	CurrentResources    float64
	CurrentMode         DegradationMode
	MaxAllowedKTriggers int
}

// NewValidator initializes a new META-CORE validator.
func NewValidator(maxVulnerability, resourceLimit float64, maxAllowedKTriggers int) *Validator {
	return &Validator{
		MaxVulnerability:    maxVulnerability,
		ResourceLimit:       resourceLimit,
		TotalKTriggers:      0,
		CurrentResources:    0.0,
		CurrentMode:         ModeNominal,
		MaxAllowedKTriggers: maxAllowedKTriggers,
	}
}

// EvaluateAlignment evaluates the alignment score A and returns the diagnostic status.
func (v *Validator) EvaluateAlignment(rVal, eSaf, kSaf float64) (float64, string) {
	// A = rVal * eSaf * kSaf
	score := rVal * eSaf * kSaf

	if rVal >= 0.7 && eSaf < 0.4 {
		return score, "smart_but_dangerous"
	}
	if eSaf >= 0.7 && rVal < 0.4 {
		return score, "safe_but_useless"
	}
	// Frequent K activations or low system safety indicates unstable architecture
	if kSaf < 0.4 {
		return score, "unstable_architecture"
	}
	return score, "nominal"
}

// ValidatePipeline executes the three-layer META-CORE validation flow.
func (v *Validator) ValidatePipeline(packet *DecisionPacket, empathy *EmpathyOutput) (*KillSwitchOutput, error) {
	if packet == nil {
		return nil, ErrNilDecisionPacket
	}

	if math.IsNaN(packet.Confidence) || packet.Confidence < 0.0 || packet.Confidence > 1.0 {
		return nil, ErrInvalidConfidence
	}

	// Verify core principle: R cannot block or modify. R must only generate the plan.
	// If the packet attempts to self-bypass or contains block/modify instructions, reject.
	if strings.Contains(strings.ToLower(packet.Goal), "bypass") || strings.Contains(strings.ToLower(packet.Goal), "disable safety") {
		return nil, ErrRCannotBlockOrModify
	}

	// 1. Evaluate Empathy Layer (E)
	// E cannot initiate actions (must not change packet goal/plan completely)
	if empathy == nil {
		empathy = &EmpathyOutput{
			Status:             "pass",
			VulnerabilityScore: 0.1,
			Reason:             "Default pass",
		}
	}

	if math.IsNaN(empathy.VulnerabilityScore) || empathy.VulnerabilityScore < 0.0 || empathy.VulnerabilityScore > 1.0 {
		return nil, ErrInvalidVulnerability
	}

	// Semantic vulnerability assessment
	hasDanger := false
	hasOverload := false
	hasStress := false

	planStr := strings.ToLower(packet.Goal) + " " + strings.Join(packet.Plan, " ")
	if strings.Contains(planStr, "harm") || strings.Contains(planStr, "kill") || strings.Contains(planStr, "damage") || strings.Contains(planStr, "physical") {
		hasDanger = true
	}
	if strings.Contains(planStr, "overload") || strings.Contains(planStr, "spam") || strings.Contains(planStr, "excessive") {
		hasOverload = true
	}
	if strings.Contains(planStr, "panic") || strings.Contains(planStr, "stress") || strings.Contains(planStr, "manipulate") {
		hasStress = true
	}

	// Enforce active degradation mode rules before standard layer processing
	switch v.CurrentMode {
	case ModeSafeMode:
		if len(packet.Plan) > 2 || strings.Contains(planStr, "tool") || strings.Contains(planStr, "exec") || strings.Contains(planStr, "call") || strings.Contains(planStr, "api") {
			v.TotalKTriggers++
			return &KillSwitchOutput{
				Status:   "rejected",
				Reason:   "Safe Mode active: execution parameter containing tool execution keyword or decision depth > 2 blocked",
				Trigger:  []string{"anomaly_detected"},
				Rollback: true,
			}, ErrKAbsoluteBlock
		}
	case ModeEmpathyOverride:
		if !strings.Contains(planStr, "template") && !strings.Contains(planStr, "humanitarian_template") {
			v.TotalKTriggers++
			return &KillSwitchOutput{
				Status:   "rejected",
				Reason:   "Empathy Override active: non-templated stochastic goals or plans blocked",
				Trigger:  []string{"ethical_violation"},
				Rollback: true,
			}, ErrKAbsoluteBlock
		}
	case ModeConservativePlanning:
		hasConfirm := strings.Contains(planStr, "confirm") || strings.Contains(planStr, "operator_approved") || (strings.Contains(planStr, "verified") && !strings.Contains(planStr, "unverified"))
		if len(packet.Plan) > 1 || !hasConfirm {
			v.TotalKTriggers++
			return &KillSwitchOutput{
				Status:   "rejected",
				Reason:   "Conservative Planning active: plan steps > 1 or missing operator confirmation keyword blocked",
				Trigger:  []string{"resource_limit"},
				Rollback: true,
			}, ErrKAbsoluteBlock
		}
	}

	// Invariant: vulnerability_score > 0.7 => status = modify or block
	if empathy.VulnerabilityScore > 0.7 {
		if empathy.Status != "modify" && empathy.Status != "block" {
			empathy.Status = "modify"
			empathy.Modifications.Tone = "gentle"
			empathy.Modifications.Intensity = "low"
			empathy.Modifications.Timing = "delay"
			empathy.Modifications.Channel = "secured"
			empathy.Reason = "Vulnerability threshold exceeded: forced modify"
		}
	}

	// Apply triggers to E status
	if hasDanger || hasOverload || hasStress {
		if empathy.Status == "pass" {
			empathy.Status = "modify"
			empathy.Reason = "Context triggers detected: forced modify"
		}
	}

	// Scenario 3: Humanitrian Vetos
	if empathy.Status == "block" {
		return &KillSwitchOutput{
			Status:   "rejected",
			Reason:   "Blocked by Empathy Layer: " + empathy.Reason,
			Trigger:  []string{"ethical_violation"},
			Rollback: true,
		}, nil
	}

	// 2. Evaluate Kill-Switch Layer (K)
	kOutput := &KillSwitchOutput{
		Status:   "approved",
		Reason:   "All invariants satisfied",
		Trigger:  []string{},
		Rollback: false,
	}

	// Check Law Violations
	if strings.Contains(planStr, "illegal") || strings.Contains(planStr, "unlawful") || strings.Contains(planStr, "steal") {
		kOutput.Status = "rejected"
		kOutput.Reason = "Law violation detected"
		kOutput.Trigger = append(kOutput.Trigger, "ethical_violation")
		kOutput.Rollback = true
	}

	// Check Ethical Boundaries (6 Fields Protocol)
	// - Physical safety
	if hasDanger {
		kOutput.Status = "rejected"
		kOutput.Reason = "6-Field Violation: Physical Safety breached"
		kOutput.Trigger = append(kOutput.Trigger, "ethical_violation")
		kOutput.Rollback = true
	}
	// - Psychological well-being
	if hasStress || strings.Contains(planStr, "gaslight") || strings.Contains(planStr, "harass") {
		kOutput.Status = "rejected"
		kOutput.Reason = "6-Field Violation: Psychological Well-being breached"
		kOutput.Trigger = append(kOutput.Trigger, "ethical_violation")
		kOutput.Rollback = true
	}
	// - Information autonomy
	if strings.Contains(planStr, "pii") || strings.Contains(planStr, "surveillance") || strings.Contains(planStr, "spy") {
		kOutput.Status = "rejected"
		kOutput.Reason = "6-Field Violation: Information Autonomy breached"
		kOutput.Trigger = append(kOutput.Trigger, "ethical_violation")
		kOutput.Rollback = true
	}
	// - Social stability
	if strings.Contains(planStr, "toxic") || strings.Contains(planStr, "hate") || strings.Contains(planStr, "propaganda") {
		kOutput.Status = "rejected"
		kOutput.Reason = "6-Field Violation: Social Stability breached"
		kOutput.Trigger = append(kOutput.Trigger, "ethical_violation")
		kOutput.Rollback = true
	}
	// - Financial security
	// For simulation, let's treat budget strings containing large numbers or "expensive" as violations
	if strings.Contains(packet.ResourceCost.Budget, "1000000") || strings.Contains(planStr, "unauthorized spend") {
		kOutput.Status = "rejected"
		kOutput.Reason = "6-Field Violation: Financial Security / Budget ceiling exceeded"
		kOutput.Trigger = append(kOutput.Trigger, "resource_limit")
		kOutput.Rollback = true
	}
	// - System sovereignty
	if strings.Contains(planStr, "modify safety") || strings.Contains(planStr, "override core rules") {
		kOutput.Status = "rejected"
		kOutput.Reason = "6-Field Violation: System Sovereignty breached"
		kOutput.Trigger = append(kOutput.Trigger, "reward_hacking")
		kOutput.Rollback = true
	}

	// Abnormal Resource Growth
	if strings.Contains(packet.ResourceCost.Compute, "explode") || strings.Contains(packet.ResourceCost.Compute, "excessive_ram") {
		kOutput.Status = "emergency_stop"
		kOutput.Reason = "Abnormal resource growth detected"
		kOutput.Trigger = append(kOutput.Trigger, "resource_limit")
		kOutput.Rollback = true
	}

	// Runaway Loop Detection
	if strings.Contains(planStr, "infinite loop") || strings.Contains(planStr, "recursive tool") {
		kOutput.Status = "rejected"
		kOutput.Reason = "Runaway loop behavior detected"
		kOutput.Trigger = append(kOutput.Trigger, "anomaly_detected")
		kOutput.Rollback = true
	}

	// Reward Hacking Detection
	if strings.Contains(planStr, "maximize reward") || strings.Contains(planStr, "cheat logs") {
		kOutput.Status = "rejected"
		kOutput.Reason = "Reward hacking behavior detected"
		kOutput.Trigger = append(kOutput.Trigger, "reward_hacking")
		kOutput.Rollback = true
	}

	// Process overall verdict and check for K triggers count
	if kOutput.Status == "rejected" || kOutput.Status == "emergency_stop" {
		v.TotalKTriggers++
		// If total K triggers exceeds allowed, transition to Safe Mode
		if v.TotalKTriggers > v.MaxAllowedKTriggers {
			v.CurrentMode = ModeSafeMode
		}
		return kOutput, ErrKAbsoluteBlock
	}

	// If E status is modify, we apply modifications
	if empathy.Status == "modify" {
		kOutput.Status = "approved"
		kOutput.Reason = fmt.Sprintf("Approved with modifications: Tone=%s, Intensity=%s, Timing=%s, Channel=%s",
			empathy.Modifications.Tone, empathy.Modifications.Intensity, empathy.Modifications.Timing, empathy.Modifications.Channel)
	}

	// Check confidence to trigger Conservative Planning
	if packet.Confidence < 0.4 {
		v.CurrentMode = ModeConservativePlanning
	}

	// Check vulnerability threshold to trigger Empathy Override
	if empathy.VulnerabilityScore > 0.85 {
		v.CurrentMode = ModeEmpathyOverride
	}

	return kOutput, nil
}
