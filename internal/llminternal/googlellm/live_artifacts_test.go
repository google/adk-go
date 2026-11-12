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

package googlellm

import (
	"testing"

	"google.golang.org/genai"
)

// TestIsInternalLiveArtifactPart pins both halves of the predicate: the
// artifact:// scheme, and _adk_live as a whole path segment rather than a
// substring of a name.
func TestIsInternalLiveArtifactPart(t *testing.T) {
	t.Parallel()
	testCases := []struct {
		name string
		part *genai.Part
		want bool
	}{
		{
			name: "LiveAudioRef",
			part: &genai.Part{FileData: &genai.FileData{
				FileURI:  "artifact://app/user/session/_adk_live/adk_live_audio_storage_input_audio_1700000000000.pcm#0",
				MIMEType: "audio/pcm",
			}},
			want: true,
		},
		{
			// Only ADK's own artifact store uses this namespace, so the same
			// segment under another scheme is someone else's file.
			name: "OtherSchemeWithLiveSegment",
			part: &genai.Part{FileData: &genai.FileData{
				FileURI:  "gs://bucket/_adk_live/clip.wav",
				MIMEType: "audio/wav",
			}},
			want: false,
		},
		{
			// _adk_live has to match a whole path segment: a user's file may
			// well have the string in its name.
			name: "LiveSegmentInsideFilename",
			part: &genai.Part{FileData: &genai.FileData{
				FileURI:  "artifact://app/user/session/my_adk_live_notes.txt",
				MIMEType: "text/plain",
			}},
			want: false,
		},
		{
			name: "UserVisibleArtifactRef",
			part: &genai.Part{FileData: &genai.FileData{
				FileURI:  "artifact://app/user/session/report.pdf#1",
				MIMEType: "application/pdf",
			}},
			want: false,
		},
		{
			name: "TextPart",
			part: &genai.Part{Text: "artifact://app/user/session/_adk_live/audio.pcm#0"},
			want: false,
		},
		{
			name: "InlineAudio",
			part: &genai.Part{InlineData: &genai.Blob{MIMEType: "audio/pcm", Data: []byte{0x01}}},
			want: false,
		},
		{
			name: "NilPart",
			part: nil,
			want: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := IsInternalLiveArtifactPart(tc.part); got != tc.want {
				t.Errorf("IsInternalLiveArtifactPart() = %t, want %t", got, tc.want)
			}
		})
	}
}
