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
// adk-python's FirestoreSessionService, so a session written by one ADK
// runtime can be read and resumed by the other, within the limits described
// under Cross-Runtime Compatibility:
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
// their numeric values round-trip as float64 (preserving exact integer
// precision up to 2^53). App and user state are stored as Firestore values.
// The values ADK Go writes also go through [encoding/json] first, so whole
// numbers are stored as integers, other numbers as doubles, and a time.Time
// or []byte as a string. Values already in the document, such as timestamps
// or bytes written natively by adk-python, keep their types when ADK Go
// updates other keys. Numbers in app and user state read back as float64 too.
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
// # Cross-Runtime Compatibility
//
// An event's "event_data" field holds the event with snake_case field names
// and timestamps as float epoch seconds, the shape adk-python dumps. Keys
// inside user data, such as state deltas, function and tool call arguments and
// responses, part metadata, and requested auth configs, are kept as written.
// [session.Service.AppendEvent] refuses an event that would not decode from
// what it stores, such as one whose requested auth config does not decode as a
// [google.golang.org/adk/v2/tool/toolauth.Config], rather than store an event
// that makes the session fail to load.
//
// The service stores "event_data" as a native map, the format adk-python
// releases through 2.13.0 write and read. Firestore rejects some maps: one
// with an empty key or a reserved one such as "__type__", one nested more
// than 20 levels deep, one with more than about 20,000 values, each of which
// Firestore indexes, or one over the size limit once each number counts 8
// bytes. The Firestore emulator and some databases also reject an array
// directly inside another array, such as a tool response
// {"rows": [["a", 1]]}. The service stores an event in any of these shapes
// as a JSON string instead, the format adk-python's main branch writes for
// every event. On
// read, the service accepts both formats, camelCase field names, and bytes
// that adk-python wrote in URL-safe base64, such as thought signatures and
// inline data.
//
// In practice, for an application that moves between adk-python and ADK Go
// on one database:
//
//   - ADK Go loads sessions written by any adk-python release or by its main
//     branch. Event fields that [session.Event] does not define, such as
//     adk-python's actions.route, agent_state, end_of_agent and
//     rewind_before_invocation_id, are not read. A session whose state holds
//     NaN or Infinity, which adk-python writes as JSON that [encoding/json]
//     does not parse, fails to load.
//   - adk-python releases through 2.13.0 read sessions written by ADK Go,
//     except a session holding an event stored as a string: adk-python reads
//     a session's events in one pass, so that one event makes the whole
//     session fail to load. Those releases cannot store such an event
//     themselves, so only sessions created or continued by ADK Go have them.
//     A release of adk-python that reads strings loads these sessions too.
//   - adk-python loads events without the fields only ADK Go defines, such as
//     [session.EventCompaction.ExcludedEvents]. Its EventActions and
//     EventCompaction reject fields they do not define, though, so a field
//     ADK Go adds to either later makes events carrying it unreadable by
//     adk-python versions that lack it.
//
// adk-python stores a compaction's range as float seconds, which keep only
// microseconds, and has no field for [session.EventCompaction.ExcludedEvents].
// The service writes both exactly under an extra "adk_go_compaction" key that
// adk-python ignores, and reads them back from it.
//
// Firestore rejects an array directly inside another array. In the app and
// user state documents, each such inner array is stored as
// {"_adk_list": [...]} and a one-key map in an array whose key is "_adk_list"
// or "_adk_dict" as {"_adk_dict": {...}}, as adk-python's main branch does,
// and both are unwrapped on read. adk-python releases through 2.13.0 neither
// write nor unwrap these, and cannot store such state themselves, so they
// return app or user state that ADK Go wrote with a nested array in its
// wrapped form.
//
// # Concurrency and Consistency
//
// Firestore reads and queries are strongly consistent, and writes execute in
// serializable transactions. A transaction that loses to a concurrent write
// is retried, up to [Config.TransactionMaxAttempts] (defaulting to the
// Firestore client default of 5 attempts), before the error is returned.
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
// sessions contend on that document and their transactions retry. Use "user:"
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
// document ID before applying [session.GetRequest.NumRecentEvents], so the tail
// window reflects insertion order even when multiple events share a timestamp.
// A positive NumRecentEvents therefore still reads every event matching
// [session.GetRequest.After].
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
// This package does not create Firestore indexes. Single-session operations
// ([session.Service.Create], [session.Service.Get],
// [session.Service.AppendEvent], [session.Service.Delete]) and per-user
// [session.Service.List] calls work with Firestore's automatic single-field
// indexes. Listing sessions across all users of an application (calling
// [session.Service.List] with an empty UserID) runs a collection-group query,
// which needs a collection-group index on the "appName" field of the sessions
// collection group. Without it Firestore fails the query with
// FailedPrecondition, and the error carries a console link that creates the
// index.
//
// A field's index configuration replaces the automatic one, and per-user
// List calls rely on the collection-scope index, so keep the collection-scope
// entries when adding the collection-group one. As a field override in
// firestore.indexes.json:
//
//	{
//	  "fieldOverrides": [{
//	    "collectionGroup": "sessions",
//	    "fieldPath": "appName",
//	    "indexes": [
//	      {"order": "ASCENDING", "queryScope": "COLLECTION"},
//	      {"order": "DESCENDING", "queryScope": "COLLECTION"},
//	      {"arrayConfig": "CONTAINS", "queryScope": "COLLECTION"},
//	      {"order": "ASCENDING", "queryScope": "COLLECTION_GROUP"}
//	    ]
//	  }]
//	}
//
// The gcloud CLI cannot create this index: "gcloud firestore indexes fields
// update" only writes collection-scope entries.
package firestore
