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

// Package platform provides seams for overriding system operations, such as
// reading the current time and generating unique IDs.
//
// By default these operations use the wall clock (time.Now) and random UUIDs.
// Users that need custom behavior can install their own providers on a
// context.Context with WithTimeProvider and WithUUIDProvider. ADK reads the
// current time and new IDs through Now and NewUUID, so an installed provider
// transparently controls the timestamps and identifiers that end up in events
// and sessions.
//
// Providers are carried explicitly on a context.Context. Carrying the
// provider on the context (rather than in a package-level variable) also
// keeps it isolated to a single call tree, which is what makes concurrent
// runs with independent providers safe.
package platform
