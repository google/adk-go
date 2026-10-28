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

package agentanalytics

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"cloud.google.com/go/storage"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"
)

const (
	// gcsUploadTimeout bounds one batch's uploads to Cloud Storage when
	// Config.GCSUploadTimeout is not set.
	gcsUploadTimeout = 10 * time.Second

	// rawPartsKey is the row key under which the plugin queues the payloads
	// that the BatchProcessor still has to upload. It is not a column.
	rawPartsKey = "_raw_parts"

	// summaryBuilderKey is the row key under which addContent attaches a
	// contentSummaryBuilder when a row has pending text uploads. Like
	// rawPartsKey, it is deleted by resolveUploadsForBatch before Arrow
	// encoding.
	summaryBuilderKey = "_summary_builder"

	// deferredUploadsKey holds a *deferredRowUploads, which
	// resolveUploadsForBatch attaches to every row with uploads. When canDefer
	// is true and the end of the caller's context, rather than the batch's own
	// upload timeout, ends the uploads of a batch that did not pause them, the
	// row's uploads that returned an error are recorded on it (none if all of
	// them succeeded), so that if the batch's subsequent BigQuery write in
	// flushRunEnd also runs out of flushCtx and the batch is deferred to the
	// next flush, that flush retries those uploads; the key then stays on the
	// row until the caller requeues the uploads (requeueDeferredUploads) or
	// clears them (clearDeferredUploads). Otherwise resolveUploadsForBatch
	// removes it itself.
	deferredUploadsKey = "_deferred_uploads"

	// mediaOffloadedText is the text of an uploaded binary part.
	mediaOffloadedText = "[MEDIA OFFLOADED]"

	// uploadFailedText is the text of a binary part whose upload failed,
	// matching adk-python.
	uploadFailedText = "[UPLOAD FAILED]"

	// textPreviewRunes and textPreviewSuffix make the preview that stays
	// inline in place of an uploaded text part.
	textPreviewRunes  = 200
	textPreviewSuffix = "... [OFFLOADED]"

	// textUploadContentType is the content type of uploaded text objects and
	// of their object_ref.details.gcs_metadata.content_type. Go differs from
	// adk-python here: adk-python uploads text as "text/plain", whereas Go
	// sets "text/plain; charset=utf-8" so HTTP readers of the Cloud Storage
	// object do not treat multi-byte UTF-8 runes as Latin-1.
	// content_parts[].mime_type remains "text/plain", as in adk-python, while
	// queries that filter on $.gcs_metadata.content_type in
	// object_ref.details see "text/plain; charset=utf-8" for Go rows and
	// "text/plain" for Python rows.
	textUploadContentType = "text/plain; charset=utf-8"

	// Object names use these in place of ids that a row does not have. The
	// event id alone keeps such names unique.
	missingTraceID = "no_trace"
	missingSpanID  = "no_span"
	missingEventID = "no_event"

	// fallbackExtension is the extension of objects whose MIME type has none.
	fallbackExtension = ".bin"

	// defaultMaxPendingUploadBytes caps total bytes held in memory by pending
	// uploads across queued rows (256 MiB).
	defaultMaxPendingUploadBytes = 256 * 1024 * 1024
)

// mimeExtensions fixes the extension of the MIME types that matter most. The
// mime package answers from the MIME tables of the host, and for some types,
// such as image/jpeg, its first extension is not the common one.
var mimeExtensions = map[string]string{
	"application/json": ".json",
	"application/pdf":  ".pdf",
	"audio/mpeg":       ".mp3",
	"audio/wav":        ".wav",
	"image/gif":        ".gif",
	"image/jpeg":       ".jpg",
	"image/png":        ".png",
	"image/webp":       ".webp",
	"text/plain":       ".txt",
	"video/mp4":        ".mp4",
	"video/webm":       ".webm",
}

// pendingUpload is a payload that is to be uploaded to Cloud Storage when the
// row it belongs to is written, and the entry of the row's content_parts that
// then refers to it. It is built on the goroutine of the plugin callback and
// owns its data, so it never shares memory with the caller's messages.
type pendingUpload struct {
	// slot is the index of the entry in the row's content_parts.
	slot int
	// contentOrd and partIdx locate the part in its request and message.
	contentOrd int
	partIdx    int
	// mimeType is the MIME type of the part.
	mimeType string
	// data is the payload of a binary part: a copy that nothing else uses.
	data []byte
	// text is the payload of a text part. A string is immutable, so it needs
	// no copy.
	text   string
	isText bool
	// fallbackTruncated reports whether the inline fallback text for this part
	// was truncated by MaxContentLen.
	fallbackTruncated bool
	// reserved is the byte count reserved in BatchProcessor.pendingUploadBytes.
	reserved int64
}

// size returns the length of the payload in bytes.
func (u pendingUpload) size() int {
	if u.isText {
		return len(u.text)
	}
	return len(u.data)
}

// contentType returns the content type of the object.
func (u pendingUpload) contentType() string {
	if u.isText {
		return textUploadContentType
	}
	return u.mimeType
}

// payload returns the bytes of the object. Text is converted on demand, so
// that a queued row holds one copy of a long text, not two.
func (u pendingUpload) payload() []byte {
	if u.isText {
		return []byte(u.text)
	}
	return u.data
}

// objectRef returns the object_ref of the object at uri. authorizer is left
// out, which makes it NULL, when it is empty. version is always left out:
// adk-python writes NULL, and an empty string would be an invalid generation.
func objectRef(uri, details, authorizer string) map[string]any {
	ref := map[string]any{"uri": uri, "details": details}
	if authorizer != "" {
		ref["authorizer"] = authorizer
	}
	return ref
}

// gcsMetadataDetails returns the details of an object_ref: the metadata of
// the uploaded object, in the form BigQuery uses for ObjectRefs. Unlike
// adk-python, which omits "size" from object_ref.details.gcs_metadata on all
// offloaded parts (both text and binary), Go records the byte length ("size")
// for both text and binary parts so readers of object_ref.details have the
// object size without a separate storage lookup.
func gcsMetadataDetails(contentType string, sizeBytes int) string {
	b, err := json.Marshal(map[string]any{
		"gcs_metadata": map[string]any{"content_type": contentType, "size": sizeBytes},
	})
	if err != nil {
		return "{}"
	}
	return string(b)
}

// textPreview is what stays inline of a text that was uploaded: its first
// textPreviewRunes runes and a note.
func textPreview(text string) string {
	n := 0
	for i := range text {
		if n == textPreviewRunes {
			text = text[:i]
			break
		}
		n++
	}
	// BigQuery rejects strings that are not valid UTF-8.
	return strings.ToValidUTF8(text, "\uFFFD") + textPreviewSuffix
}

// extensionForMIME returns the file extension, with its dot, of objects of
// the given MIME type. Parameters such as a charset are ignored.
func extensionForMIME(mimeType string) string {
	base, _, _ := strings.Cut(mimeType, ";")
	base = strings.ToLower(strings.TrimSpace(base))
	if ext, ok := mimeExtensions[base]; ok {
		return ext
	}
	if exts, err := mime.ExtensionsByType(base); err == nil {
		for _, ext := range exts {
			if isSafeExtension(ext) {
				return ext
			}
		}
	}
	return fallbackExtension
}

// isSafeExtension reports whether ext is a dot and a few ASCII letters or
// digits. The MIME tables of the host are not under our control, and an
// extension ends up in an object name.
func isSafeExtension(ext string) bool {
	if len(ext) < 2 || len(ext) > 17 || ext[0] != '.' {
		return false
	}
	for _, r := range ext[1:] {
		if (r < 'a' || r > 'z') && (r < 'A' || r > 'Z') && (r < '0' || r > '9') {
			return false
		}
	}
	return true
}

// pathSegment returns id for use as a part of an object name: id with
// anything but letters, digits, '-', '_' and '.' replaced, or fallback if id
// is empty or is nothing but dots, which would make it a relative path
// element such as "..".
func pathSegment(id, fallback string) string {
	if strings.Trim(id, ".") == "" {
		return fallback
	}
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			return r
		default:
			return '_'
		}
	}, id)
}

// gcsObjectPath returns the name of the object that holds a part:
//
//	[prefix/]<UTC date>/<trace id>/<span id>_<event id>_c<content>_p<part><ext>
//
// The date segment uses the row timestamp in UTC rather than the host's local
// date (which adk-python uses via datetime.now().date()), so object prefixes
// are independent of the server time zone and match the UTC day partition of
// the row's timestamp column. The content ordinal tells apart the messages of
// one request, whose part indexes both start at 0, and the event id tells apart
// the rows that share a span. The name is a function of its inputs, so retries
// of the same deterministic row/part path by the storage Writer itself target
// the same object.
func gcsObjectPath(prefix string, ts time.Time, traceID, spanID, eventID string, contentOrd, partIdx int, ext string) string {
	var sb strings.Builder
	if p := strings.Trim(prefix, "/"); p != "" {
		sb.WriteString(p)
		sb.WriteByte('/')
	}
	fmt.Fprintf(&sb, "%s/%s/%s_%s_c%d_p%d%s",
		ts.UTC().Format(time.DateOnly),
		pathSegment(traceID, missingTraceID),
		pathSegment(spanID, missingSpanID),
		pathSegment(eventID, missingEventID),
		contentOrd, partIdx, ext)
	return sb.String()
}

// uploadTask is one payload to upload and the content_parts entry that is
// rewritten once the upload is done. The upload goroutine writes uri and err
// and nothing else; the entry is only touched after all uploads are done.
type uploadTask struct {
	row     map[string]any
	part    map[string]any
	builder *contentSummaryBuilder
	pending pendingUpload
	path    string

	uri string
	err error
}

// uploadResolveOptions configures how resolveUploadsForBatch handles a batch's
// Cloud Storage uploads.
type uploadResolveOptions struct {
	pauseOnCallerDeadline bool
	canDefer              bool
}

// resolveUploadsForBatch uploads the payloads that were queued with the rows of
// batch and rewrites the matching content_parts entries to refer to the
// objects. No row is lost to a storage problem: a text part whose upload fails
// keeps its truncated inline text, and a binary part whose upload fails gets
// the text "[UPLOAD FAILED]", its raw bytes not being stored inline.
//
// Uploads run concurrently, at most GCSMaxConcurrentUploads at a time, under a
// single batch-level timeout that ctx can end sooner. Cloud Storage is not
// keeping up when at least one upload started and:
//
//   - the batch's own GCSUploadTimeout ended the uploads, and either an
//     in-flight upload timed out under the batch deadline or no upload in the
//     batch succeeded; or
//   - opts.pauseOnCallerDeadline is true and the deadline of ctx ended the
//     uploads first, and either no upload in the batch succeeded or the
//     deadline cut short an upload that had been running for at least three
//     quarters of the time that the batch had for its uploads, whether or not
//     other uploads in the batch succeeded. An upload that the offloader
//     failed with a deadline error of its own, such as a per-request timeout,
//     while that deadline had not passed yet was not cut short by it.
//
// The uploads are then paused for the length of the timeout, and the batches
// resolved in that time do not call the offloader but fall back at once, so
// neither a full Cloud Storage outage nor a partial brownout makes every
// batch wait in turn. A caller deadline when opts.pauseOnCallerDeadline is
// false, a caller deadline that cuts short only uploads that ran for less than
// that in a batch where another upload succeeded (such as uploads that started
// late, after waiting for a free slot behind other uploads), a ctx that is
// canceled, or a ctx that is already done before any upload starts says
// nothing about Cloud Storage and pauses nothing. The cause of the batch's
// context tells which deadline ended the uploads, and the time that an upload
// ran is counted up to that deadline, so that neither depends on how soon the
// offloader returns once the context is done.
//
// resolveUploadsForBatch attaches deferredUploadsKey to every row with uploads.
// When opts.canDefer is true, ctx has ended the uploads (its deadline passed or
// it was canceled) rather than the batch's own timeout, and the batch did not
// pause them, every upload in the batch that returned an error (whether ctx cut
// it short in flight or while it waited for a slot, or it had already failed
// with another error) is recorded on its row's deferredUploadsKey and stays
// reserved: a caller that defers the batch requeues these uploads with it
// (requeueDeferredUploads), and one that writes or drops the batch releases
// them (clearDeferredUploads). Until then, a row whose uploads all succeeded
// keeps an empty deferredUploadsKey. Otherwise resolveUploadsForBatch removes
// the key from every row and keeps none of the uploads; in particular, a batch
// that paused the uploads keeps none of them, like the batches resolved during
// the pause. An upload that returned no error but a URI that is not a gs:// URI
// fails only when it is applied, and is not kept.
// resolveUploadsForBatch does not reserve any of ctx for the caller's own work:
// a caller that has to write the batch afterwards bounds ctx accordingly.
// Nothing from the payloads is logged: only the event id, part index, byte size
// and safe error metadata of parts that failed.
func (b *BatchProcessor) resolveUploadsForBatch(ctx context.Context, opts uploadResolveOptions, batch []map[string]any) {
	type rowSummary struct {
		row     map[string]any
		builder *contentSummaryBuilder
	}
	var tasks []*uploadTask
	var summaries []rowSummary
	var reservedTotal, deferredReserved int64
	var staleDrops []deferredUploadFailure
	for i := range batch {
		row := batch[i]
		staleDrops = append(staleDrops, b.releaseDeferredRowUpload(row)...)
		builder, _ := row[summaryBuilderKey].(*contentSummaryBuilder)
		delete(row, summaryBuilderKey)
		pending, ok := row[rawPartsKey].([]pendingUpload)
		if !ok {
			continue
		}
		delete(row, rawPartsKey)
		origTruncated, _ := row["is_truncated"].(bool)
		parts, _ := row[columnContentParts].([]map[string]any)
		hasRowTasks := false
		for _, u := range pending {
			reservedTotal += u.reserved
			if u.slot < 0 || u.slot >= len(parts) {
				continue
			}
			hasRowTasks = true
			tasks = append(tasks, &uploadTask{
				row:     row,
				part:    parts[u.slot],
				builder: builder,
				pending: u,
				path:    b.objectPath(row, u),
			})
		}
		if hasRowTasks {
			row[deferredUploadsKey] = &deferredRowUploads{
				builder:              builder,
				nonDeferredTruncated: origTruncated,
			}
			if builder != nil {
				summaries = append(summaries, rowSummary{row: row, builder: builder})
			}
		}
	}
	// A normal return leaves deferred rows with the caller, which still holds
	// the batch. A panic does not: flushRunEnd has already taken the batch off
	// the queue and has not stored it, and afterRun recovers the panic. Release
	// those rows too, or their bytes stay reserved for the life of the process.
	// This defer is registered first so it still runs when the log defer panics.
	defer func() {
		if r := recover(); r != nil {
			for i := range batch {
				b.releaseDeferredRowUpload(batch[i])
			}
			b.releaseUploadBytes(reservedTotal - deferredReserved)
			panic(r)
		}
		b.releaseUploadBytes(reservedTotal - deferredReserved)
	}()
	defer func() {
		logDroppedDeferredUploads(b.config.Logger, staleDrops)
	}()

	offloader := b.config.GCSOffloader
	if len(tasks) == 0 || offloader == nil {
		b.clearDeferredUploads(batch)
		if offloader == nil {
			for _, t := range tasks {
				if t.pending.isText {
					if t.pending.fallbackTruncated {
						t.row["is_truncated"] = true
					}
					if t.builder != nil {
						fallbackText, _ := t.part["text"].(string)
						t.builder.retainInlineSummary(t.pending.contentOrd, t.pending.partIdx, fallbackText, t.pending.fallbackTruncated)
					}
				}
			}
			for _, s := range summaries {
				s.builder.applyToRow(s.row)
			}
		}
		return
	}

	uploadTimeout := cmp.Or(b.uploadTimeout, b.config.GCSUploadTimeout, gcsUploadTimeout)
	if until := b.uploadCooldownUntil.Load(); until > 0 && time.Now().UnixNano() < until {
		b.clearDeferredUploads(batch)
		for _, t := range tasks {
			t.err = errUploadCooldown
			b.applyUpload(t, false)
		}
		for _, s := range summaries {
			s.builder.applyToRow(s.row)
		}
		return
	}
	// The cause that WithTimeoutCause records tells which deadline ended the
	// uploads, so that this does not depend on when the offloaders return.
	now := time.Now()
	uploadCtx, cancel := context.WithTimeoutCause(ctx, uploadTimeout, errBatchUploadTimeout)
	defer cancel()
	// An upload that the deadline ends is a stall if it had been running for at
	// least three quarters of the time that the batch has for its uploads, which
	// ends at the earlier of the batch's own timeout and the deadline of ctx.
	// The time that it waited for a free slot before it started does not count.
	deadline, _ := uploadCtx.Deadline()
	stallAfter := max(deadline.Sub(now), 0) * 3 / 4

	limit := b.config.GCSMaxConcurrentUploads
	if limit <= 0 {
		limit = defaultGCSMaxConcurrentUploads
	}
	sem := make(chan struct{}, limit)
	var wg sync.WaitGroup
	var anyStarted, anySucceeded, anyCutShort, anyStalled atomic.Bool
	for _, t := range tasks {
		sem <- struct{}{}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			var started bool
			start := time.Now()
			t.uri, t.err, started = b.upload(uploadCtx, offloader, t)
			// Whether the deadline of uploadCtx had passed when the upload
			// returned. An offloader can fail with a deadline error of its own,
			// such as a per-request timeout, before that deadline, which then
			// did not cut the upload short.
			deadlinePassed := errors.Is(uploadCtx.Err(), context.DeadlineExceeded)
			if started {
				anyStarted.Store(true)
				if errors.Is(t.err, context.DeadlineExceeded) || (t.err != nil && deadlinePassed) {
					anyCutShort.Store(true)
					// Only an upload that the deadline cut short can have
					// stalled until it. The offloader may return a while after
					// the deadline, which is no time that the upload spent
					// running.
					if ran := min(time.Since(start), deadline.Sub(start)); deadlinePassed && ran >= stallAfter {
						anyStalled.Store(true)
					}
				}
			}
			if t.err == nil && strings.HasPrefix(t.uri, "gs://") {
				anySucceeded.Store(true)
			}
		}()
	}
	wg.Wait()

	// Pause uploads when at least one upload started and:
	//   - the batch's own GCSUploadTimeout ended the uploads and either an
	//     in-flight upload timed out under the batch deadline (brownout) or no
	//     upload in the batch succeeded, or
	//   - a caller deadline allowed to pause (opts.pauseOnCallerDeadline) ended
	//     the uploads before the batch's own timeout and either no upload in the
	//     batch succeeded or the deadline cut short an upload that had been
	//     running for at least stallAfter, even if other uploads in the batch
	//     succeeded (partial outage). An upload that failed with a deadline
	//     error of its own before the deadline passed does not count.
	ownTimeoutExpired := context.Cause(uploadCtx) == errBatchUploadTimeout
	var batchTimedOut bool
	switch {
	case ownTimeoutExpired:
		batchTimedOut = anyCutShort.Load() || !anySucceeded.Load()
	case opts.pauseOnCallerDeadline && errors.Is(uploadCtx.Err(), context.DeadlineExceeded):
		batchTimedOut = anyStalled.Load() || !anySucceeded.Load()
	}
	paused := anyStarted.Load() && batchTimedOut
	if paused {
		b.uploadCooldownUntil.Store(time.Now().Add(uploadTimeout).UnixNano())
	} else if anySucceeded.Load() {
		b.uploadCooldownUntil.Store(0)
	}

	// When opts.canDefer is true and ctx has ended the uploads, rather than the
	// batch's own timeout, every upload that returned an error in the batch is
	// kept for a retry of the batch, unless this batch paused the uploads: the
	// uploads that failed once ctx had ended them, and any that had already
	// failed, so that a custom offloader that returns an opaque error when ctx
	// ends is retried too. An upload that returned no error is not kept, even
	// if its URI is not a gs:// URI.
	keepCutShort := opts.canDefer && !ownTimeoutExpired && ctx.Err() != nil && !paused
	for _, t := range tasks {
		deferred := false
		if keepCutShort && t.err != nil {
			if d, ok := t.row[deferredUploadsKey].(*deferredRowUploads); ok {
				d.pending = append(d.pending, t.pending)
				deferredReserved += t.pending.reserved
				deferred = true
			}
		}
		if !deferred && t.err != nil && t.pending.isText && t.pending.fallbackTruncated {
			if d, ok := t.row[deferredUploadsKey].(*deferredRowUploads); ok {
				d.nonDeferredTruncated = true
			}
		}
		b.applyUpload(t, deferred)
	}
	if !keepCutShort {
		b.clearDeferredUploads(batch)
	}
	for _, s := range summaries {
		s.builder.applyToRow(s.row)
	}
}

// deferredUploadFailure is the log context for one upload deferred in this pass.
// It is not part of the queued payload.
type deferredUploadFailure struct {
	eventID string
	partIdx int
	size    int
	summary string
}

type deferredRowUploads struct {
	pending              []pendingUpload
	builder              *contentSummaryBuilder
	nonDeferredTruncated bool
	// failures is log context for uploads deferred in this pass. The retry
	// sentence is logged from requeueDeferredUploads, only when that upload
	// is actually put back on the row.
	failures []deferredUploadFailure
}

// releaseDeferredRowUpload drops one row's deferred uploads and releases their
// reserved bytes. Logging happens afterwards, so a panicking logger cannot
// keep this row's reservation.
func (b *BatchProcessor) releaseDeferredRowUpload(row map[string]any) []deferredUploadFailure {
	d, ok := row[deferredUploadsKey].(*deferredRowUploads)
	delete(row, deferredUploadsKey)
	if !ok {
		return nil
	}
	var total int64
	for _, u := range d.pending {
		total += u.reserved
	}
	b.releaseUploadBytes(total)
	return d.failures
}

func logDroppedDeferredUploads(logger Logger, failures []deferredUploadFailure) {
	if logger == nil {
		return
	}
	for _, f := range failures {
		logger.Printf("agentanalytics: GCS upload failed for event %s part %d (%d bytes): %s; keeping the part inline",
			f.eventID, f.partIdx, f.size, f.summary)
	}
}

// clearDeferredRowUploads releases one row's deferred bytes without logging.
// releaseRowUploads calls it once per row of a dropped batch, so a panicking
// log here would skip the release of the rows after it. The failures stay on
// the row, and clearDeferredUploads logs them once the batch is released.
func (b *BatchProcessor) clearDeferredRowUploads(row map[string]any) {
	if failures := b.releaseDeferredRowUpload(row); len(failures) > 0 {
		row[deferredUploadsKey] = &deferredRowUploads{failures: failures}
	}
}

func (b *BatchProcessor) clearDeferredUploads(batch []map[string]any) {
	var failures []deferredUploadFailure
	for _, row := range batch {
		failures = append(failures, b.releaseDeferredRowUpload(row)...)
	}
	logDroppedDeferredUploads(b.config.Logger, failures)
}

func (b *BatchProcessor) requeueDeferredUploads(batch []map[string]any) {
	var failures []deferredUploadFailure
	for _, row := range batch {
		d, ok := row[deferredUploadsKey].(*deferredRowUploads)
		delete(row, deferredUploadsKey)
		if !ok || len(d.pending) == 0 {
			continue
		}
		parts, _ := row[columnContentParts].([]map[string]any)
		hasText := false
		for _, u := range d.pending {
			if u.slot >= 0 && u.slot < len(parts) && !u.isText {
				parts[u.slot]["text"] = inlineDataText
			}
			if u.isText {
				hasText = true
				if d.builder != nil {
					d.builder.omitTextFromSummary(u.contentOrd, u.partIdx)
				}
			}
		}
		row[rawPartsKey] = d.pending
		if d.builder != nil {
			if hasText {
				d.builder.applyToRow(row)
			}
			row[summaryBuilderKey] = d.builder
		} else if hasText {
			row["is_truncated"] = d.nonDeferredTruncated
		}
		failures = append(failures, d.failures...)
	}
	// flushRunEnd stores the batch only after this function returns. Persist it
	// first so a panicking logger cannot drop the batch or its reservations.
	b.deferredBatch = batch
	for _, f := range failures {
		b.config.logf("agentanalytics: GCS upload failed for event %s part %d (%d bytes): %s; keeping the part inline pending retry on next flush",
			f.eventID, f.partIdx, f.size, f.summary)
	}
}

// objectPath returns the name of the object that holds the part u of row.
func (b *BatchProcessor) objectPath(row map[string]any, u pendingUpload) string {
	ts, _ := row[timestampField].(time.Time)
	if ts.IsZero() {
		ts = time.Now()
	}
	traceID, _ := row["trace_id"].(string)
	spanID, _ := row["span_id"].(string)
	eventID, _ := row[eventIDField].(string)
	return gcsObjectPath(b.config.GCSPathPrefix, ts, traceID, spanID, eventID, u.contentOrd, u.partIdx, extensionForMIME(u.mimeType))
}

type offloaderPanicError struct {
	valueType string
}

func (e offloaderPanicError) Error() string {
	return fmt.Sprintf("offloader panicked (%s)", e.valueType)
}

type ownedUploadError struct {
	err error
}

func (e ownedUploadError) Error() string {
	return e.err.Error()
}

func (e ownedUploadError) Unwrap() error {
	return e.err
}

var (
	errInvalidGCSURI  = errors.New("offloader returned a URI that is not a gs:// URI")
	errUploadCooldown = errors.New("upload skipped during timeout cooldown")

	// errBatchUploadTimeout is the cause of the context of the uploads of a
	// batch when the batch's own timeout ended them, rather than the caller's
	// context. It wraps context.DeadlineExceeded so an offloader returning
	// context.Cause(ctx) logs a deadline error.
	errBatchUploadTimeout = fmt.Errorf("batch upload timeout: %w", context.DeadlineExceeded)
)

// safeUploadErrorSummary returns diagnostic metadata for err without calling
// err.Error() on arbitrary errors from custom offloaders. Errors returned by
// the built-in gcsOffloader are wrapped in ownedUploadError and logged after
// scrubbing sensitive tokens via redactErrorMessage and stripping URL query
// strings and fragments.
func safeUploadErrorSummary(err error) string {
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return context.DeadlineExceeded.Error()
	case errors.Is(err, context.Canceled):
		return context.Canceled.Error()
	case errors.Is(err, errInvalidGCSURI):
		return errInvalidGCSURI.Error()
	case errors.Is(err, errUploadCooldown):
		return errUploadCooldown.Error()
	}
	var panicErr offloaderPanicError
	if errors.As(err, &panicErr) {
		return panicErr.Error()
	}
	var owned ownedUploadError
	if errors.As(err, &owned) {
		return stripURIQueries(redactErrorMessage(owned.Error()))
	}
	var apiErr *googleapi.Error
	if errors.As(err, &apiErr) {
		return fmt.Sprintf("HTTP %d", apiErr.Code)
	}
	return fmt.Sprintf("error type %T", err)
}

// stripURIQueries removes query parameters and fragments from URIs in msg so
// log lines carry no URLs with query parameters.
func stripURIQueries(msg string) string {
	if !strings.Contains(msg, "://") {
		return msg
	}
	return uriTokenRe.ReplaceAllStringFunc(msg, func(token string) string {
		trimmed := trimTrailingPunctuation(token)
		suffix := token[len(trimmed):]
		if i := strings.IndexAny(trimmed, "?#"); i >= 0 {
			trimmed = trimmed[:i]
		}
		return trimmed + suffix
	})
}

// upload runs one upload. The offloader may be supplied by the user, so a
// panic in it is turned into an error that names only its type. Once ctx is
// done the offloader is no longer called, which is what ends the uploads that
// were still waiting for their turn.
func (b *BatchProcessor) upload(ctx context.Context, offloader GCSContentOffloader, t *uploadTask) (uri string, err error, started bool) {
	defer func() {
		if r := recover(); r != nil {
			uri, err = "", offloaderPanicError{valueType: fmt.Sprintf("%T", r)}
		}
	}()
	if cerr := ctx.Err(); cerr != nil {
		return "", cerr, false
	}
	started = true
	uri, err = offloader.UploadContent(ctx, GCSUploadRequest{
		Path:        t.path,
		Data:        t.pending.payload(),
		ContentType: t.pending.contentType(),
	})
	return uri, err, started
}

// applyUpload rewrites the content_parts entry of a successful upload to
// refer to the object, and logs a failed one. The entry already has the MIME
// type of the part, as it was queued with the one that is uploaded.
func (b *BatchProcessor) applyUpload(t *uploadTask, deferred bool) {
	u := t.pending
	err := t.err
	if err == nil && !strings.HasPrefix(t.uri, "gs://") {
		err = errInvalidGCSURI
	}
	if err != nil {
		eventID, _ := t.row[eventIDField].(string)
		summary := safeUploadErrorSummary(err)
		if deferred {
			// Logged later: the plain failure when the upload is dropped, or
			// the retry sentence when requeueDeferredUploads puts it back.
			if d, ok := t.row[deferredUploadsKey].(*deferredRowUploads); ok {
				d.failures = append(d.failures, deferredUploadFailure{
					eventID: eventID,
					partIdx: u.partIdx,
					size:    u.size(),
					summary: summary,
				})
			}
		} else {
			b.config.logf("agentanalytics: GCS upload failed for event %s part %d (%d bytes): %s; keeping the part inline",
				eventID, u.partIdx, u.size(), summary)
		}
		if !u.isText {
			t.part["text"] = uploadFailedText
		} else {
			if u.fallbackTruncated {
				t.row["is_truncated"] = true
			}
			if t.builder != nil {
				fallbackText, _ := t.part["text"].(string)
				t.builder.retainInlineSummary(u.contentOrd, u.partIdx, fallbackText, u.fallbackTruncated)
			}
		}
		return
	}
	t.part["storage_mode"] = storageModeGCSReference
	t.part["uri"] = t.uri
	t.part["object_ref"] = objectRef(t.uri, gcsMetadataDetails(u.contentType(), u.size()), b.config.ConnectionID)
	if u.isText {
		t.part["text"] = textPreview(u.text)
		if t.builder != nil {
			t.builder.omitTextFromSummary(u.contentOrd, u.partIdx)
		}
	} else {
		t.part["text"] = mediaOffloadedText
	}
}

// gcsOffloader is the GCSContentOffloader that stores objects in a Cloud
// Storage bucket.
type gcsOffloader struct {
	client *storage.Client
	bucket *storage.BucketHandle
	name   string
}

var _ GCSContentOffloader = (*gcsOffloader)(nil)

// newGCSOffloader creates an offloader for the named bucket.
func newGCSOffloader(ctx context.Context, bucket string, opts ...option.ClientOption) (*gcsOffloader, error) {
	client, err := storage.NewClient(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create Cloud Storage client: %w", err)
	}
	return &gcsOffloader{client: client, bucket: client.Bucket(bucket), name: bucket}, nil
}

// UploadContent writes req.Data to the object req.Path and returns its gs://
// URI. The object is only created if it does not exist, so an existing
// object is never overwritten. Unlike adk-python, which fails on HTTP 412,
// an existing object is treated as success so that retries of the same
// deterministic row/part path by the storage Writer itself stay idempotent.
func (o *gcsOffloader) UploadContent(ctx context.Context, req GCSUploadRequest) (string, error) {
	w := o.bucket.Object(req.Path).If(storage.Conditions{DoesNotExist: true}).NewWriter(ctx)
	w.ContentType = req.ContentType
	// The payload is in memory already: a chunk just larger than it keeps the
	// single-request upload without the client's default 16 MiB buffer.
	if len(req.Data) < googleapi.DefaultUploadChunkSize {
		w.ChunkSize = len(req.Data) + 1
	}
	_, err := w.Write(req.Data)
	// Close also releases the resources of a writer that failed.
	if closeErr := w.Close(); err == nil {
		err = closeErr
	}
	if err != nil && !isPreconditionFailed(err) {
		return "", ownedUploadError{err: fmt.Errorf("failed to write the Cloud Storage object: %w", err)}
	}
	return "gs://" + o.name + "/" + req.Path, nil
}

// Close releases the Cloud Storage client.
func (o *gcsOffloader) Close() error {
	return o.client.Close()
}

// isPreconditionFailed reports whether err is the HTTP 412 that Cloud Storage
// answers when an object that must not exist does.
func isPreconditionFailed(err error) bool {
	var apiErr *googleapi.Error
	return errors.As(err, &apiErr) && apiErr.Code == http.StatusPreconditionFailed
}
