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
	"testing"

	"google.golang.org/genai"
)

func TestFlattenContentText(t *testing.T) {
	tests := []struct {
		name    string
		content *genai.Content
		want    string
		wantErr bool
	}{
		{
			name:    "nil content",
			content: nil,
			want:    "",
		},
		{
			name: "valid text parts",
			content: &genai.Content{
				Parts: []*genai.Part{
					{Text: "part1"},
					nil,
					{Text: "part2"},
				},
			},
			want: "part1\npart2",
		},
		{
			name: "non-text part",
			content: &genai.Content{
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{Name: "fn"}},
				},
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			txt, err := FlattenContentText(tc.content)
			if (err != nil) != tc.wantErr {
				t.Fatalf("FlattenContentText() error = %v, wantErr %v", err, tc.wantErr)
			}
			if txt != tc.want {
				t.Fatalf("FlattenContentText() = %q, want %q", txt, tc.want)
			}
		})
	}
}

// TestUnsupportedPayload_MediaIsReportedUntilAnEndpointClaimsIt pins the
// default answer, which is the thing standing between a half-migrated package
// and an image that leaves the request in silence.
//
// While one endpoint can emit media and the other cannot, the endpoint that
// cannot must keep getting the old answer. Were the default flipped instead,
// every caller that has not yet grown a wire field would start dropping media
// rather than refusing it — the failure that is invisible in review, because
// nothing errors and the model simply never sees the picture.
func TestUnsupportedPayload_MediaIsReportedUntilAnEndpointClaimsIt(t *testing.T) {
	tests := []struct {
		name string
		part *genai.Part
		want string
	}{
		{
			name: "inline_data",
			part: &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}}},
			want: "InlineData",
		},
		{
			name: "file_data",
			part: &genai.Part{FileData: &genai.FileData{MIMEType: "image/png", FileURI: "file-a"}},
			want: "FileData",
		},
		{
			// A part carrying both is reported for FileData, because the walk
			// returns the first non-zero field in genai.Part's declaration
			// order and FileData is declared ahead of InlineData there.
			//
			// That is the opposite of the order ClassifyMedia emits them in,
			// and the difference is not a defect in either: one names a field
			// to report, the other sequences content on the wire. Stated here
			// so a later reader does not "fix" one to match the other.
			name: "both_fields_report_the_one_declared_first",
			part: &genai.Part{
				InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}},
				FileData:   &genai.FileData{MIMEType: "application/pdf", FileURI: "file-b"},
			},
			want: "FileData",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := UnsupportedPayload(tt.part); got != tt.want {
				t.Errorf("UnsupportedPayload() = %q, want %q: an endpoint with no wire field for media must refuse it, not drop it", got, tt.want)
			}
			if got := UnsupportedPayload(tt.part, PartFieldMedia); got != "" {
				t.Errorf("UnsupportedPayload(PartFieldMedia) = %q, want %q", got, "")
			}
		})
	}
}

// TestUnsupportedPayload_EmittingMediaDoesNotHideAnythingElse keeps the
// exemption narrow: a part carrying media and something unsendable is still
// reported for the something, so an endpoint that gains media does not go
// quiet about the rest.
func TestUnsupportedPayload_EmittingMediaDoesNotHideAnythingElse(t *testing.T) {
	part := &genai.Part{
		InlineData:     &genai.Blob{MIMEType: "image/png", Data: []byte{1}},
		ExecutableCode: &genai.ExecutableCode{Code: "print(1)"},
	}
	if got := UnsupportedPayload(part, PartFieldMedia); got != "ExecutableCode" {
		t.Errorf("UnsupportedPayload(PartFieldMedia) = %q, want %q", got, "ExecutableCode")
	}
}

// TestReplayedReasoning_MediaStillDisqualifies states that the drop was not
// widened by the exemption: reasoning is droppable only when the part holds
// nothing else, and media is something else no matter which endpoint asks.
func TestReplayedReasoning_MediaStillDisqualifies(t *testing.T) {
	part := &genai.Part{
		Thought:    true,
		Text:       "scratch",
		InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}},
	}
	if ReplayedReasoning(part) {
		t.Error("ReplayedReasoning() = true for a thought carrying an image, want false: dropping the part would drop the image with it")
	}
}

// TestUnsupportedPayload_MediaResolutionIsLostNoLongerSilently covers the one
// field whose meaning changed when an endpoint started sending media.
//
// MediaResolution qualifies media. While the media itself was refused, the
// qualifier went with it and the caller heard about the media; now that the
// media goes out, a resolution no wire field carries would be a request
// quietly downgraded. So on a part whose media is sent it is reported, and on
// a part with no media — where it qualifies nothing — it stays accounted for.
func TestUnsupportedPayload_MediaResolutionIsLostNoLongerSilently(t *testing.T) {
	high := &genai.PartMediaResolution{Level: genai.PartMediaResolutionLevelMediaResolutionHigh}
	tests := []struct {
		name string
		part *genai.Part
		want string
	}{
		{
			name: "on_media_that_is_sent_it_is_reported",
			part: &genai.Part{
				InlineData:      &genai.Blob{MIMEType: "image/png", Data: []byte{1}},
				MediaResolution: high,
			},
			want: "MediaResolution",
		},
		{
			name: "on_file_data_that_is_sent_it_is_reported",
			part: &genai.Part{
				FileData:        &genai.FileData{MIMEType: "image/png", FileURI: "file-a"},
				MediaResolution: high,
			},
			want: "MediaResolution",
		},
		{
			// Nothing for it to qualify, so nothing is lost by dropping it.
			name: "without_media_it_qualifies_nothing",
			part: &genai.Part{Text: "hi", MediaResolution: high},
			want: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := UnsupportedPayload(tt.part, PartFieldMedia); got != tt.want {
				t.Errorf("UnsupportedPayload(PartFieldMedia) = %q, want %q", got, tt.want)
			}
			// Without the claim the media itself is reported first, exactly as
			// before: an endpoint that cannot send media never reaches the
			// question of what qualifies it.
			if tt.part.InlineData != nil || tt.part.FileData != nil {
				if got := UnsupportedPayload(tt.part); got == "MediaResolution" {
					t.Errorf("UnsupportedPayload() = %q, want the media field itself", got)
				}
			}
		})
	}
}
