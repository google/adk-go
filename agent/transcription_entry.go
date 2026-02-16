package agent

import "google.golang.org/genai"

type TranscriptionEntry struct {
	// Role that created this data, typically "user" or "model". For function call, this is None.
	Role string

	// Data is the content of the transcription entry, which can be either a Blob or a Content.
	Data TranscriptionEntryData
}

type TranscriptionEntryData struct {
	Blob    *genai.Blob
	Content *genai.Content
}
