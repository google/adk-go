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

package feature

import (
	"testing"
)

func TestRegisterAndDefault(t *testing.T) {
	reset()
	MustRegister("FOO", Config{Stage: Stable, DefaultOn: true})
	if !IsEnabled("FOO") {
		t.Error("default-on stable feature should be enabled")
	}
	MustRegister("BAR", Config{Stage: Experimental, DefaultOn: false})
	if IsEnabled("BAR") {
		t.Error("default-off experimental feature should be disabled")
	}
}

func TestOverridePriority(t *testing.T) {
	reset()
	MustRegister("FOO", Config{Stage: Experimental, DefaultOn: false})
	if IsEnabled("FOO") {
		t.Fatal("expected disabled by default")
	}
	if err := Override("FOO", true); err != nil {
		t.Fatalf("Override err: %v", err)
	}
	if !IsEnabled("FOO") {
		t.Error("override should win")
	}
	ClearOverride("FOO")
	if IsEnabled("FOO") {
		t.Error("override cleared, should fall back to default")
	}
}

func TestEnvVar(t *testing.T) {
	reset()
	MustRegister("FOO", Config{Stage: Experimental, DefaultOn: false})
	t.Setenv("ADK_ENABLE_FOO", "true")
	if !IsEnabled("FOO") {
		t.Error("ADK_ENABLE_FOO should enable")
	}
	t.Setenv("ADK_ENABLE_FOO", "")
	t.Setenv("ADK_DISABLE_FOO", "1")
	if IsEnabled("FOO") {
		t.Error("ADK_DISABLE_FOO should disable")
	}
}

func TestOverrideBeatsEnv(t *testing.T) {
	reset()
	MustRegister("FOO", Config{Stage: Experimental, DefaultOn: false})
	t.Setenv("ADK_ENABLE_FOO", "true")
	_ = Override("FOO", false)
	if IsEnabled("FOO") {
		t.Error("programmatic override should beat env")
	}
}

func TestWithOverrideRestore(t *testing.T) {
	reset()
	MustRegister("FOO", Config{Stage: Experimental, DefaultOn: false})
	restore := WithOverride("FOO", true)
	if !IsEnabled("FOO") {
		t.Fatal("with-override should enable")
	}
	restore()
	if IsEnabled("FOO") {
		t.Error("restore should drop override")
	}
}

func TestWithOverrideRestoresPrior(t *testing.T) {
	reset()
	MustRegister("FOO", Config{Stage: Experimental, DefaultOn: false})
	_ = Override("FOO", false) // explicit prior override
	restore := WithOverride("FOO", true)
	if !IsEnabled("FOO") {
		t.Fatal("inner override should take effect")
	}
	restore()
	if IsEnabled("FOO") {
		t.Error("restore should put back the prior explicit override (false)")
	}
}

// TestWithOverrideRestoresPriorFalse_NotDefault distinguishes "key
// existed in overrides with value false" from "key did not exist". A
// previous bug used the bool value as the existence flag, which made
// restoring an explicit-false override identical to deleting the entry
// (and therefore falling through to the registry default). This test
// would fail under that bug because the registry default is true.
func TestWithOverrideRestoresPriorFalse_NotDefault(t *testing.T) {
	reset()
	MustRegister("FOO", Config{Stage: Stable, DefaultOn: true})
	if err := Override("FOO", false); err != nil {
		t.Fatalf("Override: %v", err)
	}
	restore := WithOverride("FOO", true)
	if !IsEnabled("FOO") {
		t.Fatal("inner override (true) should take effect")
	}
	restore()
	if IsEnabled("FOO") {
		t.Errorf("restore should put back the explicit-false override; " +
			"got true (looks like the override was deleted and the registry default leaked through)")
	}
}

func TestCheck(t *testing.T) {
	reset()
	MustRegister("FOO", Config{Stage: Stable, DefaultOn: true})
	if err := Check("FOO"); err != nil {
		t.Errorf("Check should pass for enabled: %v", err)
	}
	MustRegister("BAR", Config{Stage: Experimental, DefaultOn: false})
	if err := Check("BAR"); err == nil {
		t.Error("Check should error for disabled")
	}
}

func TestUnregisteredPanics(t *testing.T) {
	reset()
	defer func() {
		if recover() == nil {
			t.Error("expected panic on unregistered feature")
		}
	}()
	IsEnabled("DOES_NOT_EXIST")
}

func TestRegisterStageMismatchErrors(t *testing.T) {
	reset()
	MustRegister("FOO", Config{Stage: Experimental, DefaultOn: true})
	if err := Register("FOO", Config{Stage: Stable, DefaultOn: true}); err == nil {
		t.Error("expected error redeclaring with different stage")
	}
}
