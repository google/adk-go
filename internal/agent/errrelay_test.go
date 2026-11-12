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

package agent

import (
	"errors"
	"testing"
)

func TestErrRelay(t *testing.T) {
	boom := errors.New("boom")
	var got []error
	ctx := WithErrRelay(t.Context(), "inv", "seq", func(err error) { got = append(got, err) })

	// Only the agent the relay is bound to, in its invocation, reports.
	ReportErr(ctx, "inv", "inner", boom)
	ReportErr(ctx, "other", "seq", boom)
	if len(got) != 0 {
		t.Fatalf("ReportErr() through a relay bound elsewhere reported %v, want nothing", got)
	}

	// Only the bound agent can pass the relay on.
	if _, ok := PassErrRelay(ctx, "inv", "catcher", "inner"); ok {
		t.Error("PassErrRelay() from an agent the relay is not bound to = true, want false")
	}
	if _, ok := PassErrRelay(ctx, "other", "seq", "inner"); ok {
		t.Error("PassErrRelay() in another invocation = true, want false")
	}
	innerCtx, ok := PassErrRelay(ctx, "inv", "seq", "inner")
	if !ok {
		t.Fatal("PassErrRelay() from the bound agent = false, want true")
	}

	ReportErr(innerCtx, "inv", "seq", boom)
	if len(got) != 0 {
		t.Fatalf("ReportErr() by the agent that passed the relay on reported %v, want nothing", got)
	}
	ReportErr(innerCtx, "inv", "inner", boom)
	if len(got) != 1 || got[0] != boom {
		t.Errorf("ReportErr() by the agent the relay was passed to reported %v, want [%v]", got, boom)
	}
}
