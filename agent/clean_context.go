// // Copyright 2026 Google LLC
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// package agent

// import (
// 	"context"
// 	"time"

// 	"google.golang.org/adk/session"
// 	"google.golang.org/adk/tool/toolconfirmation"
// )

// // clean context should provide:
// //   - pure golang-context functionalities with stacking of contexts
// //   - well define adk-context operation with shallow / deep copies on demand
// //   - descendant context creation with params controlling what to copy
// //   - minimal public API

// // in general, CleanContext should
// //  - be based on InvocationContext in terms of underlying data - create new CleanInvocationContext based on InvocationContext
// //  - provide a way to create
// //      CleanInvocationContext => CleanToolContext
// //      CleanInvocationContext => CleanCallbackContext
// //      CleanInvocationContext => CleanNodeContext
// //      CleanInvocationContext => CleanDynamicNodeContext
// //      CleanToolContext => CleanToolContext
// //      CleanToolContext => CleanCallbackContext
// //      CleanToolContext => CleanNodeContext
// //      CleanToolContext => CleanDynamicNodeContext
// //      CleanCallbackContext => CleanToolContext
// //      CleanCallbackContext => CleanCallbackContext
// //      CleanCallbackContext => CleanNodeContext
// //      CleanCallbackContext => CleanDynamicNodeContext
// //      CleanNodeContext => CleanToolContext
// //      CleanNodeContext => CleanCallbackContext
// //      CleanNodeContext => CleanNodeContext
// //      CleanNodeContext => CleanDynamicNodeContext
// //      CleanDynamicNodeContext => CleanToolContext
// //      CleanDynamicNodeContext => CleanCallbackContext
// //      CleanDynamicNodeContext => CleanNodeContext
// //      CleanDynamicNodeContext => CleanDynamicNodeContext

// func NewCleanContext(ctx context.Context) cleanContext {
// 	return cleanContext{
// 		ctx: ctx,
// 	}
// }

// type cleanContext struct {
// 	ctx    context.Context
// 	adkCtx adkCleanContext
// }

// type adkCleanContext struct {
// 	artifacts Artifacts
// 	actions   *session.EventActions

// 	// Fields below are only populated by NewToolContext.
// 	functionCallID   string
// 	toolConfirmation *toolconfirmation.ToolConfirmation

// 	// Fields below are used by NodeContext
// 	// resumeInputs are keyed by InterruptID. Nil on fresh activations
// 	// and on handoff resume.
// 	resumeInputs map[string]any

// 	// path and runID are populated for dynamic children, empty for
// 	// top-level static activations.
// 	path  string
// 	runID string

// 	// subScheduler is non-nil only when this context belongs to a
// 	// dynamic-node activation; RunNode uses it to schedule children.
// 	subScheduler DynamicSubScheduler

// 	// outputForAncestors are the delegating-ancestor paths carried
// 	// into this activation when it runs as a WithUseAsOutput child;
// 	// its dynamic sub-scheduler reads them to stamp OutputFor.
// 	outputForAncestors []string
// }

// var _ context.Context = (*cleanContext)(nil)

// // Deadline implements [context.Context].
// func (c *cleanContext) Deadline() (deadline time.Time, ok bool) {
// 	return c.ctx.Deadline()
// }

// // Done implements [context.Context].
// func (c *cleanContext) Done() <-chan struct{} {
// 	return c.ctx.Done()
// }

// // Err implements [context.Context].
// func (c *cleanContext) Err() error {
// 	return c.ctx.Err()
// }

// // Value implements [context.Context].
// func (c *cleanContext) Value(key any) any {
// 	return c.ctx.Value(key)
// }

// // WithValue returns shallow copy of original cleanContext
// // The underlying ctx is enriched with WithValue
// func (c *cleanContext) WithValue(key, val any) cleanContext {
// 	res := *c
// 	res.ctx = context.WithValue(c.ctx, key, val)
// 	return res
// }

// // // WithContext sets the underlying golang context.
// // // Should be used only with pure golang contexts
// // // does not accept
// // func (c *cleanContext) WithContext(ctx context.Context) cleanContext {

// // 	res := *c
// // 	res.ctx = ctx
// // 	return res
// // }
