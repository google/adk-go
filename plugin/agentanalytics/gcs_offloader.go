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
	// rawPartsKey, it is deleted by resolveUploads before Arrow encoding.
	summaryBuilderKey = "_summary_builder"

	// mediaOffloadedText is the text of an uploaded binary part.
	mediaOffloadedText = "[MEDIA OFFLOADED]"

	// uploadFailedText is the text of a binary part whose upload failed,
	// matching adk-python.
	uploadFailedText = "[UPLOAD FAILED]"

	// textPreviewRunes and textPreviewSuffix make the preview that stays
	// inline in place of an uploaded text part.
	textPreviewRunes  = 200
	textPreviewSuffix = "... [OFFLOADED]"

	// textUploadContentType is the content type of uploaded text objects.
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
// the uploaded object, in the form BigQuery uses for ObjectRefs.
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
// The content ordinal tells apart the messages of one request, whose part
// indexes both start at 0, and the event id tells apart the rows that share a
// span. The name is a function of its inputs, so retries of the same
// deterministic row/part path by the storage Writer itself target the same object.
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

// resolveUploads uploads the payloads that were queued with the rows of batch
// and rewrites the matching content_parts entries to refer to the objects. A
// part whose upload fails keeps the inline form it was queued with, so no
// event is lost to a storage problem.
//
// Uploads run concurrently, at most GCSMaxConcurrentUploads at a time, under a
// single batch-level timeout. Nothing from the payloads is logged: only the
// event id, part index, byte size and safe error metadata of parts that failed.
func (b *BatchProcessor) resolveUploads(ctx context.Context, batch []map[string]any) {
	type rowSummary struct {
		row     map[string]any
		builder *contentSummaryBuilder
	}
	var tasks []*uploadTask
	var summaries []rowSummary
	var reservedTotal int64
	for _, row := range batch {
		builder, _ := row[summaryBuilderKey].(*contentSummaryBuilder)
		delete(row, summaryBuilderKey)
		pending, ok := row[rawPartsKey].([]pendingUpload)
		if !ok {
			continue
		}
		delete(row, rawPartsKey)
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
		if hasRowTasks && builder != nil {
			summaries = append(summaries, rowSummary{row: row, builder: builder})
		}
	}
	defer b.releaseUploadBytes(reservedTotal)

	offloader := b.config.GCSOffloader
	if len(tasks) == 0 || offloader == nil {
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

	// Leave the BigQuery write at least half of any caller or shutdown budget.
	if deadline, ok := ctx.Deadline(); ok {
		var cancelDeadline context.CancelFunc
		ctx, cancelDeadline = context.WithDeadline(ctx, time.Now().Add(time.Until(deadline)/2))
		defer cancelDeadline()
	}
	uploadTimeout := cmp.Or(b.uploadTimeout, b.config.GCSUploadTimeout, gcsUploadTimeout)
	if until := b.uploadCooldownUntil.Load(); until > 0 && time.Now().UnixNano() < until {
		for _, t := range tasks {
			t.err = errUploadCooldown
			b.applyUpload(t)
		}
		for _, s := range summaries {
			s.builder.applyToRow(s.row)
		}
		return
	}
	parentCtx := ctx
	ctx, cancel := context.WithTimeout(parentCtx, uploadTimeout)
	defer cancel()

	limit := b.config.GCSMaxConcurrentUploads
	if limit <= 0 {
		limit = defaultGCSMaxConcurrentUploads
	}
	sem := make(chan struct{}, limit)
	var wg sync.WaitGroup
	for _, t := range tasks {
		sem <- struct{}{}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			t.uri, t.err = b.upload(ctx, offloader, t)
		}()
	}
	wg.Wait()

	if errors.Is(ctx.Err(), context.DeadlineExceeded) && parentCtx.Err() == nil {
		b.uploadCooldownUntil.Store(time.Now().Add(uploadTimeout).UnixNano())
	} else {
		for _, t := range tasks {
			if t.err == nil {
				b.uploadCooldownUntil.Store(0)
				break
			}
		}
	}

	for _, t := range tasks {
		b.applyUpload(t)
	}
	for _, s := range summaries {
		s.builder.applyToRow(s.row)
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
)

// safeUploadErrorSummary returns diagnostic metadata for err without calling
// err.Error() on arbitrary errors from custom offloaders. Errors returned by
// the built-in gcsOffloader are wrapped in ownedUploadError and logged after
// scrubbing sensitive tokens via redactErrorMessage.
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
		return redactErrorMessage(owned.Error())
	}
	var apiErr *googleapi.Error
	if errors.As(err, &apiErr) {
		return fmt.Sprintf("HTTP %d", apiErr.Code)
	}
	return fmt.Sprintf("error type %T", err)
}

// upload runs one upload. The offloader may be supplied by the user, so a
// panic in it is turned into an error that names only its type. Once ctx is
// done the offloader is no longer called, which is what ends the uploads that
// were still waiting for their turn.
func (b *BatchProcessor) upload(ctx context.Context, offloader GCSContentOffloader, t *uploadTask) (uri string, err error) {
	defer func() {
		if r := recover(); r != nil {
			uri, err = "", offloaderPanicError{valueType: fmt.Sprintf("%T", r)}
		}
	}()
	if cerr := ctx.Err(); cerr != nil {
		return "", cerr
	}
	return offloader.UploadContent(ctx, GCSUploadRequest{
		Path:        t.path,
		Data:        t.pending.payload(),
		ContentType: t.pending.contentType(),
	})
}

// applyUpload rewrites the content_parts entry of a successful upload to
// refer to the object, and logs a failed one. The entry already has the MIME
// type of the part, as it was queued with the one that is uploaded.
func (b *BatchProcessor) applyUpload(t *uploadTask) {
	u := t.pending
	err := t.err
	if err == nil && !strings.HasPrefix(t.uri, "gs://") {
		err = errInvalidGCSURI
	}
	if err != nil {
		eventID, _ := t.row[eventIDField].(string)
		b.config.logf("agentanalytics: GCS upload failed for event %s part %d (%d bytes): %s; keeping the part inline",
			eventID, u.partIdx, u.size(), safeUploadErrorSummary(err))
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
