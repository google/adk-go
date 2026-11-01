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

// Package firestore provides a [session.Service] implementation backed by
// Google Cloud Firestore.
//
// # Storage Layout
//
// The service stores session documents, event documents, and shared state
// documents using the same collection hierarchy and field layout as
// adk-python's FirestoreSessionService, allowing sessions written by one ADK
// runtime to be read and resumed by another:
//
//	{rootCollection}/{appName}/users/{userID}/sessions/{sessionID}
//	{rootCollection}/{appName}/users/{userID}/sessions/{sessionID}/events/{eventID}
//	{appStateCollection}/{appName}
//	{userStateCollection}/{appName}/users/{userID}
//
// Session-scoped state is stored as a JSON object string on the session
// document (and accepted as either a JSON string or a map on read). App-scoped
// ("app:") and user-scoped ("user:") state entries are stored without their
// scope prefixes in their respective shared documents and merged back with
// prefixes on read. Temporary ("temp:") state keys are applied to the
// in-memory session for the current invocation and stripped before persistence.
// On [session.Service.AppendEvent], session-scoped state deltas are merged onto
// the session state read inside the Firestore transaction (matching
// session/database rather than adk-python's in-memory snapshot), so direct
// mutations via [session.State.Set] outside [session.EventActions.StateDelta]
// affect only the in-memory handle and are not persisted. Because session state
// and event payloads are marshaled through [encoding/json] into map[string]any,
// numeric values round-trip as float64 (preserving exact integer precision up
// to 2^53).
//
// Session timestamps ("createTime" and "updateTime") and event timestamps are
// truncated to microsecond precision to match Cloud Firestore's storage
// resolution. Session timestamps are stamped from
// [google.golang.org/adk/v2/platform.Now] on [session.Service.Create] and from
// the appended event's Timestamp on [session.Service.AppendEvent] (assigning
// platform.Now when Event.Timestamp is zero). This matches the in-memory and
// database session services in adk-go so [session.Session.LastUpdateTime] is
// identical on the in-memory handle and on subsequent [session.Service.Get] or
// [session.Service.List] calls, whereas adk-python uses Firestore's
// SERVER_TIMESTAMP sentinel. Both timestamp representations are accepted on
// read.
//
// # Concurrency and Consistency
//
// Firestore reads and queries are strongly consistent, and writes execute in
// serializable optimistic transactions. Unlike SQL backends that block on
// pessimistic row locks (SELECT ... FOR UPDATE), Firestore aborts and retries
// contending transactions up to [Config.TransactionMaxAttempts] (defaulting to
// the Firestore client default of 5 attempts) before returning an error.
//
// Each session document maintains a monotonic integer revision counter
// incremented on every [session.Service.AppendEvent] call. If the stored
// revision does not match the revision observed when the caller's
// [session.Session] handle was created or loaded, AppendEvent rejects the write
// with [ErrStaleSession]. Within a single process, concurrent AppendEvent calls
// targeting the same session are serialized with a context-aware per-session
// lock that aborts waiting callers if their [context.Context] is canceled.
//
// Because all users of an application share a single "{appStateCollection}/{appName}"
// document, high-frequency writes to "app:" state keys across many concurrent
// sessions contend on that document (Firestore sustains roughly one write per
// second to a single document before transaction retries increase). Use "user:"
// or session-scoped keys for frequently mutated state.
//
// [session.Service.Delete] marks the session document with status "DELETING" in
// a transaction before streaming and deleting its event subcollection via
// [cloud.google.com/go/firestore.BulkWriter]. Sessions in the "DELETING" state
// are hidden from [session.Service.Get] (returning [session.ErrNotFound]) and
// [session.Service.List], and reject concurrent [session.Service.AppendEvent]
// calls. Delete is idempotent: if a deletion is interrupted after marking the
// session, calling Delete again removes any remaining events and the session
// document, and operators can query for session documents with
// status == "DELETING" to sweep incomplete deletions.
//
// [session.Service.Get] orders events by "timestamp" ascending on the server
// and tie-breaks equal timestamps in memory by the event's session revision and
// document ID. When [session.GetRequest.NumRecentEvents] is positive,
// Firestore's LimitToLast window selects the tail by (timestamp, document ID)
// before the client-side revision tie-break is applied.
//
// # Document Size Limit
//
// Firestore enforces a 1 MiB (1,048,576-byte) limit per document, including
// field names, document path, and Protobuf encoding overhead. This package
// rejects serialized session state or event JSON payloads exceeding 960 KiB
// (1 MiB minus 64 KiB of headroom for document metadata). Events carrying large
// inline binary payloads (such as [google.golang.org/genai.Blob]) should store
// binary media through an artifact service and reference it via
// [google.golang.org/genai.FileData] URIs instead.
//
// # Required Indexes
//
// This package does not create Firestore indexes automatically. Single-session
// operations ([session.Service.Create], [session.Service.Get],
// [session.Service.AppendEvent], [session.Service.Delete]) and per-user
// [session.Service.List] calls work with Firestore's automatic single-field
// indexes. Listing sessions across all users of an application (calling
// [session.Service.List] with an empty UserID) requires a collection-group
// index on the sessions collection for the "appName" field:
//
//	gcloud firestore indexes fields update appName \
//	  --collection-group=sessions \
//	  --index=order=ASCENDING,query-scope=COLLECTION_GROUP
package firestore
