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

// Package vertexairag provides a [memory.Service] backed by Vertex AI RAG corpora.
package vertexairag

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	"google.golang.org/api/option"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
)

const (
	sourceDisplayNamePrefix       = "adk-memory-v1."
	ragFilePageSize         int32 = 100
	// Scoping walks the corpus file list, so it is capped to keep the cost of a
	// search independent of how large a shared corpus grows.
	maxRAGFilePages                = 10
	defaultVectorDistanceThreshold = 10.0
)

// Config configures a Vertex AI RAG memory service.
type Config struct {
	// RAGCorpus is the Vertex AI RAG corpus resource name
	// ("projects/{project}/locations/{location}/ragCorpora/{rag_corpus_id}")
	// or bare corpus ID ("{rag_corpus_id}").
	RAGCorpus string
	// SimilarityTopK is the maximum number of contexts to retrieve.
	// A zero value leaves top-k unset on the retrieval query.
	SimilarityTopK int32
	// VectorDistanceThreshold filters out contexts whose vector distance
	// exceeds this threshold. A zero value defaults to 10.
	VectorDistanceThreshold float64
	// Project is the Google Cloud project ID for the RAG corpus.
	// If empty, GOOGLE_CLOUD_PROJECT is used, falling back to the project
	// parsed from a fully-qualified RAGCorpus resource name.
	Project string
	// Location is the Google Cloud location for the RAG corpus.
	// If empty, GOOGLE_CLOUD_LOCATION is used, falling back to the location
	// parsed from a fully-qualified RAGCorpus resource name.
	Location string
}

// NewService creates a [memory.Service] backed by Vertex AI RAG.
//
// The service reaches Vertex AI over HTTP, because RAG file uploads require
// HTTP media upload (the protobuf UploadRagFileRequest has no field for the
// file bytes), so opts are applied to HTTP clients only. Options that need a
// gRPC transport, such as [option.WithGRPCConn], are not supported.
func NewService(ctx context.Context, cfg Config, opts ...option.ClientOption) (memory.Service, error) {
	ctx = context.WithoutCancel(ctx)
	resolved := resolveConfig(cfg, os.Getenv)
	backend, err := newVertexRAGClient(ctx, resolved.Location, opts...)
	if err != nil {
		return nil, fmt.Errorf("vertexairag.NewService: %w", err)
	}
	return newServiceWithBackend(resolved, backend), nil
}

// resolveConfig fills in the fields of cfg that are left to defaults, reading
// environment variables through getenv.
func resolveConfig(cfg Config, getenv func(string) string) Config {
	if cfg.Project == "" {
		cfg.Project = getenv("GOOGLE_CLOUD_PROJECT")
	}
	if cfg.Location == "" {
		cfg.Location = getenv("GOOGLE_CLOUD_LOCATION")
	}
	if (cfg.Project == "" || cfg.Location == "") && strings.HasPrefix(cfg.RAGCorpus, "projects/") {
		parts := strings.Split(cfg.RAGCorpus, "/")
		if len(parts) >= 4 && parts[0] == "projects" && parts[2] == "locations" {
			if cfg.Project == "" {
				cfg.Project = parts[1]
			}
			if cfg.Location == "" {
				cfg.Location = parts[3]
			}
		}
	}
	// The Vertex AI gRPC and REST APIs require a full resource name for parent
	// parameters, so bare corpus IDs are expanded when project and location are known.
	if cfg.RAGCorpus != "" && !strings.HasPrefix(cfg.RAGCorpus, "projects/") && cfg.Project != "" && cfg.Location != "" {
		cfg.RAGCorpus = fmt.Sprintf("projects/%s/locations/%s/ragCorpora/%s", cfg.Project, cfg.Location, cfg.RAGCorpus)
	}
	if cfg.VectorDistanceThreshold == 0 {
		cfg.VectorDistanceThreshold = defaultVectorDistanceThreshold
	}
	return cfg
}

type service struct {
	backend                 ragBackend
	project                 string
	location                string
	similarityTopK          int32
	vectorDistanceThreshold float64
	ragResources            []*aiplatformpb.RetrieveContextsRequest_VertexRagStore_RagResource
}

var _ memory.Service = (*service)(nil)

// newServiceWithBackend creates a service from a config that resolveConfig
// has already resolved.
func newServiceWithBackend(cfg Config, backend ragBackend) *service {
	return &service{
		backend:                 backend,
		project:                 cfg.Project,
		location:                cfg.Location,
		similarityTopK:          cfg.SimilarityTopK,
		vectorDistanceThreshold: cfg.VectorDistanceThreshold,
		ragResources: []*aiplatformpb.RetrieveContextsRequest_VertexRagStore_RagResource{
			{RagCorpus: cfg.RAGCorpus},
		},
	}
}

type transcriptEvent struct {
	Author    string  `json:"author"`
	Timestamp float64 `json:"timestamp"`
	Text      string  `json:"text"`
}

// AddSessionToMemory implements [memory.Service].
func (s *service) AddSessionToMemory(ctx context.Context, curSession session.Session) error {
	if curSession == nil {
		return errors.New("session must not be nil")
	}
	if len(s.ragResources) == 0 {
		return errors.New("rag_corpus must be set on every RAG resource")
	}
	corpusNames := make([]string, 0, len(s.ragResources))
	for _, r := range s.ragResources {
		if r == nil || r.GetRagCorpus() == "" {
			return errors.New("rag_corpus must be set on every RAG resource")
		}
		corpusNames = append(corpusNames, r.GetRagCorpus())
	}

	var outputLines []string
	// adk-go has no session rewind events, unlike adk-python's _apply_rewinds, so events are processed directly.
	if events := curSession.Events(); events != nil {
		for event := range events.All() {
			if event == nil || event.Content == nil || len(event.Content.Parts) == 0 {
				continue
			}
			var textParts []string
			for _, part := range event.Content.Parts {
				if part == nil || part.Thought || part.Text == "" {
					continue
				}
				textParts = append(textParts, strings.ReplaceAll(part.Text, "\n", " "))
			}
			if len(textParts) == 0 {
				continue
			}
			var ts float64
			if !event.Timestamp.IsZero() {
				ts = float64(event.Timestamp.UnixNano()) / 1e9
			}
			line, err := marshalTranscriptEvent(transcriptEvent{
				Author:    event.Author,
				Timestamp: ts,
				Text:      strings.Join(textParts, "."),
			})
			if err != nil {
				return fmt.Errorf("marshal session event: %w", err)
			}
			outputLines = append(outputLines, line)
		}
	}

	transcript := strings.Join(outputLines, "\n")
	displayName := buildSourceDisplayName(curSession.AppName(), curSession.UserID(), curSession.ID())
	for _, corpusName := range corpusNames {
		if err := s.backend.uploadFile(ctx, corpusName, displayName, strings.NewReader(transcript)); err != nil {
			return fmt.Errorf("upload session transcript to %q: %w", corpusName, err)
		}
	}
	return nil
}

// marshalTranscriptEvent encodes ev as one JSON line. HTML escaping is off so
// that "<", ">" and "&" are indexed as written, not as \u003c escapes.
func marshalTranscriptEvent(ev transcriptEvent) (string, error) {
	var buf strings.Builder
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(ev); err != nil {
		return "", err
	}
	return strings.TrimSuffix(buf.String(), "\n"), nil
}

// SearchMemory implements [memory.Service].
func (s *service) SearchMemory(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error) {
	if req == nil {
		return nil, errors.New("search request must not be nil")
	}
	if len(s.ragResources) == 0 {
		return nil, errors.New("rag resources must be set")
	}

	scopedResources, scoped, err := s.scopedRAGResources(ctx, req.AppName, req.UserID)
	if err != nil {
		// Library code in adk-go must not log to a process-wide sink, unlike adk-python's logger.warning, so listing failures silently fall back to unscoped retrieval with response-side filtering.
		scoped = false
	}

	resources := s.ragResources
	if scoped {
		if len(scopedResources) == 0 {
			return &memory.SearchResponse{}, nil
		}
		resources = scopedResources
	}

	retrievalCfg := &aiplatformpb.RagRetrievalConfig{
		TopK: s.similarityTopK,
		Filter: &aiplatformpb.RagRetrievalConfig_Filter{
			VectorDbThreshold: &aiplatformpb.RagRetrievalConfig_Filter_VectorDistanceThreshold{
				VectorDistanceThreshold: s.vectorDistanceThreshold,
			},
		},
	}

	var parent string
	if s.project != "" && s.location != "" {
		parent = fmt.Sprintf("projects/%s/locations/%s", s.project, s.location)
	}

	resp, err := s.backend.retrieveContexts(ctx, &aiplatformpb.RetrieveContextsRequest{
		Parent: parent,
		DataSource: &aiplatformpb.RetrieveContextsRequest_VertexRagStore_{
			VertexRagStore: &aiplatformpb.RetrieveContextsRequest_VertexRagStore{
				RagResources: resources,
			},
		},
		Query: &aiplatformpb.RagQuery{
			Query:              &aiplatformpb.RagQuery_Text{Text: req.Query},
			RagRetrievalConfig: retrievalCfg,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("retrieve contexts: %w", err)
	}

	var sessionOrder []string
	sessionEvents := make(map[string][][]parsedEvent)
	for _, ragCtx := range resp.GetContexts().GetContexts() {
		if ragCtx == nil {
			continue
		}
		sourceApp, sourceUser, sessionID, ok := parseSourceDisplayName(ragCtx.GetSourceDisplayName())
		if !ok || sourceApp != req.AppName || sourceUser != req.UserID {
			continue
		}
		var events []parsedEvent
		if text := ragCtx.GetText(); text != "" {
			for line := range strings.SplitSeq(text, "\n") {
				line = strings.TrimSpace(line)
				if line == "" {
					continue
				}
				ev, ok := parseTranscriptLine(line)
				if !ok {
					continue
				}
				events = append(events, ev)
			}
		}
		if _, exists := sessionEvents[sessionID]; !exists {
			sessionOrder = append(sessionOrder, sessionID)
		}
		sessionEvents[sessionID] = append(sessionEvents[sessionID], events)
	}

	var memories []memory.Entry
	for _, sessionID := range sessionOrder {
		for _, events := range mergeEventLists(sessionEvents[sessionID]) {
			slices.SortStableFunc(events, func(a, b parsedEvent) int {
				return cmp.Compare(a.timestampSeconds, b.timestampSeconds)
			})
			for _, ev := range events {
				memories = append(memories, memory.Entry{
					Author:    ev.author,
					Content:   ev.content,
					Timestamp: floatSecondsToTime(ev.timestampSeconds),
				})
			}
		}
	}

	return &memory.SearchResponse{Memories: memories}, nil
}

func (s *service) scopedRAGResources(ctx context.Context, appName, userID string) ([]*aiplatformpb.RetrieveContextsRequest_VertexRagStore_RagResource, bool, error) {
	var scoped []*aiplatformpb.RetrieveContextsRequest_VertexRagStore_RagResource
	for _, r := range s.ragResources {
		corpus := r.GetRagCorpus()
		if corpus == "" {
			return nil, false, nil
		}
		var fileIDs []string
		var pageToken string
		for range maxRAGFilePages {
			files, nextToken, err := s.backend.listFiles(ctx, &aiplatformpb.ListRagFilesRequest{
				Parent:    corpus,
				PageSize:  ragFilePageSize,
				PageToken: pageToken,
			})
			if err != nil {
				return nil, false, err
			}
			for _, f := range files {
				if f == nil || f.GetName() == "" {
					continue
				}
				fApp, fUser, _, ok := parseSourceDisplayName(f.GetDisplayName())
				if !ok || fApp != appName || fUser != userID {
					continue
				}
				name := f.GetName()
				if idx := strings.LastIndexByte(name, '/'); idx >= 0 {
					name = name[idx+1:]
				}
				fileIDs = append(fileIDs, name)
			}
			pageToken = nextToken
			if pageToken == "" {
				break
			}
		}
		if pageToken != "" {
			// Library code in adk-go must not log to a process-wide sink, unlike adk-python's logger.warning, so hitting the 10-page cap silently falls back to unscoped retrieval.
			return nil, false, nil
		}
		if len(fileIDs) > 0 {
			scoped = append(scoped, &aiplatformpb.RetrieveContextsRequest_VertexRagStore_RagResource{
				RagCorpus:  corpus,
				RagFileIds: fileIDs,
			})
		}
	}
	return scoped, true, nil
}

func encodeSourceDisplayNamePart(value string) string {
	return base64.RawURLEncoding.EncodeToString([]byte(value))
}

func decodeSourceDisplayNamePart(value string) (string, bool) {
	value = strings.TrimRight(value, "=")
	b, err := base64.RawURLEncoding.Strict().DecodeString(value)
	if err != nil || !utf8.Valid(b) {
		return "", false
	}
	return string(b), true
}

func buildSourceDisplayName(appName, userID, sessionID string) string {
	return sourceDisplayNamePrefix +
		encodeSourceDisplayNamePart(appName) + "." +
		encodeSourceDisplayNamePart(userID) + "." +
		encodeSourceDisplayNamePart(sessionID)
}

func parseSourceDisplayName(displayName string) (appName, userID, sessionID string, ok bool) {
	if rest, found := strings.CutPrefix(displayName, sourceDisplayNamePrefix); found {
		parts := strings.Split(rest, ".")
		if len(parts) != 3 {
			return "", "", "", false
		}
		app, ok := decodeSourceDisplayNamePart(parts[0])
		if !ok {
			return "", "", "", false
		}
		user, ok := decodeSourceDisplayNamePart(parts[1])
		if !ok {
			return "", "", "", false
		}
		sess, ok := decodeSourceDisplayNamePart(parts[2])
		if !ok {
			return "", "", "", false
		}
		return app, user, sess, true
	}

	// Legacy display names were dot-delimited. Only the exact three-part form is
	// unambiguous, so dotted app/user/session IDs are intentionally ignored.
	parts := strings.Split(displayName, ".")
	if len(parts) != 3 {
		return "", "", "", false
	}
	return parts[0], parts[1], parts[2], true
}

type parsedEvent struct {
	author           string
	timestampSeconds float64
	content          *genai.Content
}

func parseTranscriptLine(line string) (parsedEvent, bool) {
	var raw map[string]any
	if err := json.Unmarshal([]byte(line), &raw); err != nil || raw == nil {
		return parsedEvent{}, false
	}
	var author string
	if a, ok := raw["author"]; ok && a != nil {
		s, ok := a.(string)
		if !ok {
			return parsedEvent{}, false
		}
		author = s
	}
	var ts float64
	if t, ok := raw["timestamp"]; ok && t != nil {
		switch v := t.(type) {
		case float64:
			ts = v
		case string:
			parsed, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
			if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) {
				return parsedEvent{}, false
			}
			ts = parsed
		default:
			return parsedEvent{}, false
		}
	}
	var text string
	if tx, ok := raw["text"]; ok && tx != nil {
		s, ok := tx.(string)
		if !ok {
			return parsedEvent{}, false
		}
		text = s
	}
	return parsedEvent{
		author:           author,
		timestampSeconds: ts,
		content: &genai.Content{
			Parts: []*genai.Part{{Text: text}},
		},
	}, true
}

func floatSecondsToTime(ts float64) time.Time {
	sec, frac := math.Modf(ts)
	return time.Unix(int64(sec), int64(math.Round(frac*1e9)))
}

func mergeEventLists(eventLists [][]parsedEvent) [][]parsedEvent {
	var merged [][]parsedEvent
	queue := slices.Clone(eventLists)
	for len(queue) > 0 {
		current := slices.Clone(queue[0])
		queue = queue[1:]

		currentTS := make(map[float64]struct{}, len(current))
		for _, ev := range current {
			currentTS[ev.timestampSeconds] = struct{}{}
		}

		mergeFound := true
		for mergeFound {
			mergeFound = false
			var remaining [][]parsedEvent
			for _, other := range queue {
				overlaps := false
				for _, ev := range other {
					if _, exists := currentTS[ev.timestampSeconds]; exists {
						overlaps = true
						break
					}
				}
				if overlaps {
					// New events are chosen against the timestamps from before
					// this merge, so events in other that share a new timestamp
					// are all kept.
					var newEvents []parsedEvent
					for _, ev := range other {
						if _, exists := currentTS[ev.timestampSeconds]; !exists {
							newEvents = append(newEvents, ev)
						}
					}
					current = append(current, newEvents...)
					for _, ev := range newEvents {
						currentTS[ev.timestampSeconds] = struct{}{}
					}
					mergeFound = true
				} else {
					remaining = append(remaining, other)
				}
			}
			queue = remaining
		}
		merged = append(merged, current)
	}
	return merged
}
