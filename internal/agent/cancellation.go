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
	"context"
	"errors"
)

// IsCallerCancellation reports whether err is the caller's cancellation of
// ctx: ctx is done and err is, or wraps, context.Canceled,
// context.DeadlineExceeded or ctx's cancellation cause. Such an error is not a
// failure to report to the error callbacks, as adk-python treats
// asyncio.CancelledError. errors.Is cannot tell an echo of the cause from an
// unrelated failure that wraps it, so a caller that cancels with a cause
// should use an error value it owns. A cancellation or deadline error raised
// inside the run while ctx is still live is not the caller's, and it reports
// false for it.
func IsCallerCancellation(ctx context.Context, err error) bool {
	return ctx.Err() != nil && (errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Cause(ctx)))
}
