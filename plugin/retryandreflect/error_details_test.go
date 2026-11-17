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

package retryandreflect

import (
	"errors"
	"strings"
	"testing"

	"google.golang.org/adk/v2/internal/toolinternal"
)

// The guidance for a tool error names the error's type before its text, as
// adk-python's "<ExceptionType>: <message>" does.
func TestRetryAndReflect_RaisedErrorGuidanceNamesTheType(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want string
	}{
		{
			name: "returned error",
			err:  errors.New("order not found"),
			want: "*errors.errorString: order not found",
		},
		{
			name: "recovered panic",
			err:  toolinternal.NewPanicError("test-tool", "secret", []byte("goroutine 1")),
			want: "panic: Invoking `test-tool()` failed with an internal error.",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &retryAndReflect{
				maxRetries:            1,
				scope:                 Invocation,
				scopedFailureCounters: make(map[string]map[string]int),
			}
			ctx := &mockContext{invocationID: "inv1"}
			tl := &mockTool{name: "test-tool"}

			res, _ := r.onToolError(ctx, tl, nil, tc.err)
			reflection, _ := res["reflection_guidance"].(string)
			if want := "**Error Details:**\n```\n" + tc.want + "\n```"; !strings.Contains(reflection, want) {
				t.Errorf("reflection guidance lacks %q:\n%s", want, reflection)
			}

			res, _ = r.onToolError(ctx, tl, nil, tc.err)
			exceeded, _ := res["reflection_guidance"].(string)
			if want := "**Last Error:**\n```\n" + tc.want + "\n```"; !strings.Contains(exceeded, want) {
				t.Errorf("exceed guidance lacks %q:\n%s", want, exceeded)
			}
		})
	}
}
