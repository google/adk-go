// Copyright 2025 Google LLC
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

package shared

import (
	"encoding/base64"
	"fmt"
	"strings"

	"google.golang.org/genai"
)

// MediaKind names what a part's media is, as its MIME type reports it. It
// answers "what is this", never "which field carries it": the endpoints differ
// on the second question and agree on the first.
type MediaKind string

const (
	// MediaKindImage is a MIME type in the image/ tree.
	MediaKindImage MediaKind = "image"
	// MediaKindAudio is a MIME type in the audio/ tree.
	MediaKindAudio MediaKind = "audio"
	// MediaKindVideo is a MIME type in the video/ tree.
	MediaKindVideo MediaKind = "video"
	// MediaKindFile is any other MIME type, application/pdf being the one the
	// endpoints have a field for.
	MediaKindFile MediaKind = "file"
)

// MediaSource names where the bytes are. It varies independently of
// [MediaKind]: an image can arrive inline, as an uploaded file id or as a URL,
// and which of those it is decides the wire field an endpoint reaches for even
// when the kind does not change.
type MediaSource string

const (
	// MediaSourceData is raw bytes carried in the request itself.
	MediaSourceData MediaSource = "data"
	// MediaSourceFileID is a file already uploaded to the provider, named by
	// the id it was given.
	MediaSourceFileID MediaSource = "file_id"
	// MediaSourceURL is a location the provider is expected to fetch.
	MediaSourceURL MediaSource = "url"
)

// fileIDPrefix is the shape of an OpenAI file id. A genai.FileData URI with
// this prefix names an upload rather than a location to fetch, and both
// endpoints have a dedicated field for it.
const fileIDPrefix = "file-"

// Media is a part's InlineData or FileData reduced to plain values: what it
// is, where it is, and the two labels a request needs to carry it. It holds no
// openai-go types, because the two endpoints disagree on those and agree on
// this.
type Media struct {
	// Kind is what the MIME type says the media is.
	Kind MediaKind
	// Source is where the bytes are.
	Source MediaSource
	// MIMEType is the type as the part declared it, parameters and all, for an
	// endpoint to send or to name in a rejection.
	MIMEType string
	// Filename is the part's display name, empty when it had none. Chat
	// Completions requires one alongside inline file data; Responses does not.
	Filename string

	// Data is the raw bytes, set only when Source is [MediaSourceData].
	Data []byte
	// FileID is the provider's id for an earlier upload, set only when Source
	// is [MediaSourceFileID].
	FileID string
	// URL is the location to fetch, set only when Source is [MediaSourceURL].
	URL string
}

// DataURL renders inline bytes as the data URL both endpoints accept in place
// of a location. It is built here rather than in each endpoint so that the
// encoding has one definition and one test.
//
// It returns "" for media that is not inline, there being no bytes to render.
func (m Media) DataURL() string {
	if m.Source != MediaSourceData {
		return ""
	}
	return fmt.Sprintf("data:%s;base64,%s", m.MIMEType, base64.StdEncoding.EncodeToString(m.Data))
}

// ClassifyMedia reduces a part's media fields to [Media] values, in the order
// a request must carry them: InlineData before FileData, since a part holding
// both presents them in that order and the content list has to keep it.
//
// It returns nil for a part carrying no media. It rejects media it cannot
// label rather than guessing one: a missing MIME type used to become
// application/octet-stream, which told the server the bytes were opaque when
// in fact nobody had looked.
//
// Deciding whether a labelled kind is sendable is left to the caller, because
// that is the half the two endpoints answer differently.
func ClassifyMedia(part *genai.Part) ([]Media, error) {
	if part == nil {
		return nil, nil
	}
	var out []Media
	if blob := part.InlineData; blob != nil {
		m, err := classifyInlineData(blob)
		if err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	if fd := part.FileData; fd != nil {
		m, err := classifyFileData(fd)
		if err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, nil
}

// classifyInlineData labels a blob of raw bytes.
func classifyInlineData(blob *genai.Blob) (Media, error) {
	if blob.MIMEType == "" {
		return Media{}, fmt.Errorf("%w: inline data", ErrMediaMIMETypeRequired)
	}
	if len(blob.Data) == 0 {
		return Media{}, fmt.Errorf("%w: inline data of type %s", ErrMediaDataRequired, blob.MIMEType)
	}
	return Media{
		Kind:     mediaKindOf(blob.MIMEType),
		Source:   MediaSourceData,
		MIMEType: blob.MIMEType,
		Filename: blob.DisplayName,
		Data:     blob.Data,
	}, nil
}

// classifyFileData labels media held somewhere else, splitting an uploaded
// file id from a location to fetch. The split is made on the URI alone: an
// image uploaded to the provider is still a file id, and sending it as a
// location would ask the server to fetch a string that is not one.
func classifyFileData(fd *genai.FileData) (Media, error) {
	if fd.FileURI == "" {
		return Media{}, fmt.Errorf("%w: file data", ErrMediaURIRequired)
	}
	if fd.MIMEType == "" {
		return Media{}, fmt.Errorf("%w: file data %s", ErrMediaMIMETypeRequired, fd.FileURI)
	}
	m := Media{
		Kind:     mediaKindOf(fd.MIMEType),
		MIMEType: fd.MIMEType,
		Filename: fd.DisplayName,
	}
	if strings.HasPrefix(fd.FileURI, fileIDPrefix) {
		m.Source = MediaSourceFileID
		m.FileID = fd.FileURI
		return m, nil
	}
	m.Source = MediaSourceURL
	m.URL = fd.FileURI
	return m, nil
}

// mediaKindOf reads the kind off a MIME type's top-level type, normalising the
// case and the surrounding whitespace a sender may vary without changing what
// the media is.
//
// Parameters after the subtype need no stripping: they can only follow the
// subtype, so the prefix test below never reaches them.
func mediaKindOf(mimeType string) MediaKind {
	essence := strings.ToLower(strings.TrimSpace(mimeType))
	switch {
	case strings.HasPrefix(essence, "image/"):
		return MediaKindImage
	case strings.HasPrefix(essence, "audio/"):
		return MediaKindAudio
	case strings.HasPrefix(essence, "video/"):
		return MediaKindVideo
	default:
		return MediaKindFile
	}
}
