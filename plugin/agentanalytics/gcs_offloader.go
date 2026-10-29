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
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

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

	// maxGCSObjectNameBytes is Cloud Storage's 1,024-byte UTF-8 object name
	// limit. maxGCSPrefixBytes and maxPathSegmentBytes bound the prefix and
	// individual identifier segments so assembled object names stay within
	// maxGCSObjectNameBytes.
	maxGCSObjectNameBytes    = 1024
	maxGCSPrefixBytes        = 512
	maxPathSegmentBytes      = 128
	pathSegmentHashSuffixLen = 65 // "~" + 64 lowercase hex digits (256-bit SHA-256)

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

// pathSegmentHashSuffix returns a deterministic 65-byte "~<64 hex digits>"
// suffix containing the full 256-bit SHA-256 digest of s. The leading "~"
// lies outside the unchanged safe identifier alphabet ([A-Za-z0-9._-]), so
// hashed outputs occupy a separate namespace from unchanged identifiers.
func pathSegmentHashSuffix(s string) string {
	sum := sha256.Sum256([]byte(s))
	return fmt.Sprintf("~%x", sum)
}

// truncateWithHash cuts base to at most maxBytes-pathSegmentHashSuffixLen
// bytes on a UTF-8 rune boundary and appends pathSegmentHashSuffix(raw), so
// raw inputs that sanitize or truncate to the same base still get distinct
// results of at most maxBytes bytes. The suffix holds the full 256-bit
// SHA-256 digest, which gives about 128-bit collision resistance. maxBytes
// must be at least pathSegmentHashSuffixLen: callers pass
// maxPathSegmentBytes, maxGCSPrefixBytes, maxGCSObjectNameBytes, or
// maxGCSObjectNameBytes less the length of an extension that gcsObjectPath
// has checked is short enough. base must be valid UTF-8: the cut keeps it
// valid but does not make it so, while raw may be any bytes.
func truncateWithHash(base, raw string, maxBytes int) string {
	suffix := pathSegmentHashSuffix(raw)
	limit := maxBytes - len(suffix)
	if len(base) > limit {
		for limit > 0 && !utf8.RuneStart(base[limit]) {
			limit--
		}
		base = base[:limit]
	}
	return base + suffix
}

// pathSegment returns id for use as a part of an object name. An id that is
// at most maxPathSegmentBytes bytes of ASCII letters, digits, '-', '_' and
// '.', and is not only dots, is returned unchanged. An empty id returns
// fallback, which must not be empty, and an id of only dots returns fallback
// followed by the "~<sha256>" suffix of id. Any other id has each rune
// outside that set (including '~') and each byte that is not valid UTF-8
// replaced with '_', and is cut to fit maxPathSegmentBytes with the
// "~<sha256>" suffix of the raw id appended.
func pathSegment(id, fallback string) string {
	if id == "" {
		return fallback
	}
	if strings.Trim(id, ".") == "" {
		return truncateWithHash(fallback, id, maxPathSegmentBytes)
	}
	var sb strings.Builder
	sb.Grow(min(len(id), maxPathSegmentBytes))
	altered := false
	for _, r := range id {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			sb.WriteRune(r)
		default:
			sb.WriteByte('_')
			altered = true
		}
	}
	sanitized := sb.String()
	if !altered && len(sanitized) <= maxPathSegmentBytes {
		return sanitized
	}
	return truncateWithHash(sanitized, id, maxPathSegmentBytes)
}

// isUnsafeGCSPrefixRune reports whether r is replaced in a GCSPathPrefix
// segment. Of these runes Cloud Storage only rejects CR and LF in object
// names, and it recommends avoiding the control characters that XML 1.0 does
// not allow; the rest is a portability choice. Control characters (U+0000
// to U+001F and U+007F to U+009F) and the Unicode line and paragraph
// separators are replaced so names stay printable on one line, '\' because
// some tools and hosts treat it as a path separator, and '~' because it
// marks the hash suffix, so a prefix kept verbatim never looks like a
// rewritten one. Spaces, '=' and other Unicode characters are kept.
func isUnsafeGCSPrefixRune(r rune) bool {
	return r < 0x20 || (r >= 0x7f && r <= 0x9f) ||
		r == '\u0085' || r == '\u2028' || r == '\u2029' ||
		r == '\\' || r == '~'
}

// cleanGCSPrefix returns the object name prefix for a configured
// GCSPathPrefix, following the contract documented on Config.GCSPathPrefix.
// Leading and trailing slashes are trimmed first, so "/logs/" and "logs"
// share a prefix. The trimmed prefix is returned verbatim unless it is
// longer than maxGCSPrefixBytes; has an empty ("//"), "." or ".." segment, a
// byte that is not valid UTF-8, or a rune for which isUnsafeGCSPrefixRune
// reports true; or starts with ".well-known/acme-challenge", which Cloud
// Storage reserves. That last check is positional and case-insensitive: it
// looks at the first two segments left once empty, "." and ".." segments are
// dropped. Any other prefix has those segments dropped, those bytes and runes
// replaced with '_', and a leading ".well-known" before "acme-challenge"
// renamed "_well-known", and is cut to fit maxGCSPrefixBytes with the
// "~<sha256>" suffix of the trimmed prefix appended, so distinct trimmed
// prefixes do not collapse onto the same object prefix.
func cleanGCSPrefix(prefix string) string {
	trimmed := strings.Trim(prefix, "/")
	if trimmed == "" {
		return ""
	}
	parts := strings.Split(trimmed, "/")
	segs := make([]string, 0, len(parts))
	altered := false
	for _, rawSeg := range parts {
		if rawSeg == "" || rawSeg == "." || rawSeg == ".." {
			altered = true
			continue
		}
		var segSB strings.Builder
		segSB.Grow(len(rawSeg))
		for i := 0; i < len(rawSeg); {
			r, width := utf8.DecodeRuneInString(rawSeg[i:])
			if (r == utf8.RuneError && width == 1) || isUnsafeGCSPrefixRune(r) {
				segSB.WriteByte('_')
				altered = true
			} else {
				segSB.WriteRune(r)
			}
			i += width
		}
		segs = append(segs, segSB.String())
	}
	if len(segs) >= 2 && strings.EqualFold(segs[0], ".well-known") && strings.EqualFold(segs[1], "acme-challenge") {
		segs[0] = "_well-known"
		altered = true
	}
	if len(segs) == 0 {
		return truncateWithHash("_", trimmed, maxGCSPrefixBytes)
	}
	joined := strings.Join(segs, "/")
	if !altered && len(joined) <= maxGCSPrefixBytes {
		return joined
	}
	return truncateWithHash(joined, trimmed, maxGCSPrefixBytes)
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
//
// The extension is part of the name as is only if it is empty or
// isSafeExtension accepts it, as every extension that extensionForMIME returns
// is. For any other extension, which only a direct call can pass, the name is
// never returned as is, however short: it gets the "~<sha256>" suffix of the
// whole name, as a name over maxGCSObjectNameBytes does. So '~' marks a
// rewritten name in the extension too, and an extension that carries the
// suffix of a rewritten name cannot make a name equal to that one.
//
// Such an extension may also be invalid UTF-8, which the prefix and the ids
// never are. The name is then valid UTF-8 too: its readable part has each run
// of invalid bytes replaced with '_', while the hash still covers the name
// with the extension as it was, so extensions that differ only in those bytes
// give distinct names.
func gcsObjectPath(prefix string, ts time.Time, traceID, spanID, eventID string, contentOrd, partIdx int, ext string) string {
	var sb strings.Builder
	if p := cleanGCSPrefix(prefix); p != "" {
		sb.WriteString(p)
		sb.WriteByte('/')
	}
	fmt.Fprintf(&sb, "%s/%s/%s_%s_c%d_p%d%s",
		ts.UTC().Format(time.DateOnly),
		pathSegment(traceID, missingTraceID),
		pathSegment(spanID, missingSpanID),
		pathSegment(eventID, missingEventID),
		contentOrd, partIdx, ext)
	out := sb.String()
	// cleanGCSPrefix and pathSegment bound their results, so even with the
	// longest int ordinals (20 bytes each) and an extension that
	// isSafeExtension accepts (at most 17 bytes), out is at most 971 bytes.
	// extensionForMIME returns only such extensions, so the rest of this
	// function is a backstop for an extension passed in directly, and the
	// branch that keeps a safe extension only matters if those bounds ever
	// grow.
	//
	// A name is returned as is only if its extension is empty or safe. The
	// name of any other extension ends in the hash of out, however short it
	// is, so that '~' marks a rewritten name in the extension as it does in
	// the ids and the prefix: an extension that holds the "~<sha256>" suffix
	// of a rewritten name cannot make a name equal to that one.
	if len(out) <= maxGCSObjectNameBytes && (ext == "" || isSafeExtension(ext)) {
		return out
	}
	if isSafeExtension(ext) && len(ext) < maxGCSObjectNameBytes-pathSegmentHashSuffixLen {
		return truncateWithHash(out[:len(out)-len(ext)], out, maxGCSObjectNameBytes-len(ext)) + ext
	}
	// Only an unsafe extension can make out invalid UTF-8: cleanGCSPrefix and
	// pathSegment return valid UTF-8, and a safe extension is ASCII. The
	// readable part of the name is made valid, and the hash is still that of
	// the raw out.
	return truncateWithHash(strings.ToValidUTF8(out, "_"), out, maxGCSObjectNameBytes)
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
// The failure of a deferred upload is not logged yet; clearDeferredUploads and
// requeueDeferredUploads log it once it is known whether the upload is
// retried. A write that panics before the caller settles the deferred uploads
// leaves them to dropDeferredUploadsQuietly, which the caller defers. The
// failure of an upload that is not deferred is logged at once.
func (b *BatchProcessor) resolveUploadsForBatch(ctx context.Context, opts uploadResolveOptions, batch []map[string]any) {
	type rowSummary struct {
		row     map[string]any
		builder *contentSummaryBuilder
	}
	var tasks []*uploadTask
	var summaries []rowSummary
	var reservedTotal, deferredReserved int64
	// Deferred uploads that a caller left unsettled are dropped before the
	// loop takes over the reservations of the queued payloads, so that a
	// Logger that panics on their failures cannot leak any of those.
	b.clearDeferredUploads(batch)
	for i := range batch {
		row := batch[i]
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
	var completedNormally bool
	defer func() {
		if !completedNormally {
			for _, row := range batch {
				delete(row, deferredUploadsKey)
			}
			b.releaseUploadBytes(reservedTotal)
		} else {
			b.releaseUploadBytes(reservedTotal - deferredReserved)
		}
	}()

	offloader := b.config.GCSOffloader
	if len(tasks) == 0 || offloader == nil {
		for _, row := range batch {
			delete(row, deferredUploadsKey)
		}
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
		completedNormally = true
		return
	}

	uploadTimeout := cmp.Or(b.uploadTimeout, b.config.GCSUploadTimeout, gcsUploadTimeout)
	if until := b.uploadCooldownUntil.Load(); until > 0 && time.Now().UnixNano() < until {
		for _, row := range batch {
			delete(row, deferredUploadsKey)
		}
		for _, t := range tasks {
			t.err = errUploadCooldown
			b.applyUpload(t, false)
		}
		for _, s := range summaries {
			s.builder.applyToRow(s.row)
		}
		completedNormally = true
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
			t.uri, started, t.err = b.upload(uploadCtx, offloader, t)
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
				eventID, _ := t.row[eventIDField].(string)
				d.pending = append(d.pending, deferredUpload{
					upload:     t.pending,
					eventID:    eventID,
					errSummary: safeUploadErrorSummary(t.err),
				})
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
	completedNormally = true
}

// deferredUpload is a failed upload of a batch whose uploads returned after
// the caller's context had ended, by an expired deadline or a cancellation,
// without pausing uploads across flushes: either one that the end of the
// context cut short or skipped, or one that had already failed with another
// error while a sibling upload kept the batch running.
// Its failure is logged only once it is known whether the upload is retried:
// when clearDeferredUploads drops it, or when requeueDeferredUploads queues it
// again for the next flush.
type deferredUpload struct {
	upload     pendingUpload
	eventID    string
	errSummary string
}

// deferredRowUploads is what deferredUploadsKey holds: the deferred uploads of
// a row, and what requeueDeferredUploads needs to put the row back in the form
// it was queued in.
type deferredRowUploads struct {
	pending              []deferredUpload
	builder              *contentSummaryBuilder
	nonDeferredTruncated bool
}

// clearDeferredUploads drops the deferred uploads of the rows of batch, whose
// parts keep the inline form that resolveUploadsForBatch gave them, and logs
// their failures as final. The reservations of all the rows are released before
// anything is logged, so that a panicking Logger cannot leak any of them.
func (b *BatchProcessor) clearDeferredUploads(batch []map[string]any) {
	b.logDeferredUploadFailures(b.dropDeferredUploadsQuietly(batch), false)
}

// dropDeferredUploadsQuietly drops the deferred uploads of the rows of batch
// and releases their reservations without logging anything, and returns the
// dropped uploads. A caller that writes a batch with deferred uploads defers
// it, so that a write that panics, as a panicking Logger makes it do, before
// the caller settles the deferred uploads cannot leak their reservations. On
// a batch whose deferred uploads are settled, it does nothing.
func (b *BatchProcessor) dropDeferredUploadsQuietly(batch []map[string]any) []deferredUpload {
	var dropped []deferredUpload
	var total int64
	for _, row := range batch {
		d, ok := row[deferredUploadsKey].(*deferredRowUploads)
		delete(row, deferredUploadsKey)
		if !ok {
			continue
		}
		for _, du := range d.pending {
			total += du.upload.reserved
		}
		dropped = append(dropped, d.pending...)
	}
	b.releaseUploadBytes(total)
	return dropped
}

// requeueDeferredUploads queues the deferred uploads of the rows of batch
// again, with their parts back in the form they were queued in, so that the
// next flush of batch retries them; the uploads keep their reservations. Their
// failures are logged as pending that retry, but only once every row has been
// requeued, so that a panicking Logger cannot leave a row without its uploads.
func (b *BatchProcessor) requeueDeferredUploads(batch []map[string]any) {
	var requeued []deferredUpload
	for _, row := range batch {
		d, ok := row[deferredUploadsKey].(*deferredRowUploads)
		delete(row, deferredUploadsKey)
		if !ok || len(d.pending) == 0 {
			continue
		}
		parts, _ := row[columnContentParts].([]map[string]any)
		pending := make([]pendingUpload, 0, len(d.pending))
		hasText := false
		for _, du := range d.pending {
			u := du.upload
			if u.slot >= 0 && u.slot < len(parts) && !u.isText {
				parts[u.slot]["text"] = inlineDataText
			}
			if u.isText {
				hasText = true
				if d.builder != nil {
					d.builder.omitTextFromSummary(u.contentOrd, u.partIdx)
				}
			}
			pending = append(pending, u)
		}
		row[rawPartsKey] = pending
		if d.builder != nil {
			if hasText {
				d.builder.applyToRow(row)
			}
			row[summaryBuilderKey] = d.builder
		} else if hasText {
			row["is_truncated"] = d.nonDeferredTruncated
		}
		requeued = append(requeued, d.pending...)
	}
	b.logDeferredUploadFailures(requeued, true)
}

// logDeferredUploadFailures logs the failures of deferred uploads that were
// dropped or, if retry is true, requeued.
func (b *BatchProcessor) logDeferredUploadFailures(uploads []deferredUpload, retry bool) {
	for _, du := range uploads {
		b.logUploadFailure(du.eventID, du.upload, du.errSummary, retry)
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
func (b *BatchProcessor) upload(ctx context.Context, offloader GCSContentOffloader, t *uploadTask) (uri string, started bool, err error) {
	defer func() {
		if r := recover(); r != nil {
			uri, err = "", offloaderPanicError{valueType: fmt.Sprintf("%T", r)}
		}
	}()
	if cerr := ctx.Err(); cerr != nil {
		return "", false, cerr
	}
	started = true
	uri, err = offloader.UploadContent(ctx, GCSUploadRequest{
		Path:        t.path,
		Data:        t.pending.payload(),
		ContentType: t.pending.contentType(),
	})
	return uri, started, err
}

// applyUpload rewrites the content_parts entry of a successful upload to refer
// to the object, or records the fallback state of a failed upload on the row
// and part. When err == nil and uri is a gs:// URI, the inline payload is
// replaced with the GCS reference: the part carries storage_mode, uri, and
// object_ref, a text part keeps a textPreview, and the summary drops the text
// part (the entry already has the MIME type of the part, as it was queued with
// the one that is uploaded). Otherwise the failure is logged at once unless the
// upload is deferred (whether a deferred upload is retried is only known later,
// so its failure is logged when clearDeferredUploads drops it or
// requeueDeferredUploads queues it again), and the fallback state is recorded:
// for a binary part the part text becomes [UPLOAD FAILED]; for a text part the
// inline text is kept, the summary retains it via retainInlineSummary, and
// is_truncated is set to true when fallbackTruncated indicates the inline
// fallback was itself truncated.
func (b *BatchProcessor) applyUpload(t *uploadTask, deferred bool) {
	u := t.pending
	err := t.err
	if err == nil && !strings.HasPrefix(t.uri, "gs://") {
		err = errInvalidGCSURI
	}
	if err != nil {
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
		if !deferred {
			eventID, _ := t.row[eventIDField].(string)
			b.logUploadFailure(eventID, u, safeUploadErrorSummary(err), false)
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

// logUploadFailure logs that the upload of part u of the row of eventID
// failed, with its byte size and errSummary but nothing from its payload.
// retry reports whether the upload is retried on the next flush.
func (b *BatchProcessor) logUploadFailure(eventID string, u pendingUpload, errSummary string, retry bool) {
	if retry {
		b.config.logf("agentanalytics: GCS upload failed for event %s part %d (%d bytes): %s; keeping the part inline pending retry on next flush",
			eventID, u.partIdx, u.size(), errSummary)
		return
	}
	b.config.logf("agentanalytics: GCS upload failed for event %s part %d (%d bytes): %s; keeping the part inline",
		eventID, u.partIdx, u.size(), errSummary)
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
