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

// Package duckduckgotool provides a web search tool using DuckDuckGo.
//
// This is an OpenAI-compatible alternative to Gemini's native GoogleSearch tool.
// Unlike GoogleSearch which runs inside the Gemini model, DuckDuckGoSearch makes
// actual HTTP requests to DuckDuckGo and works with any LLM through the standard
// function call mechanism (Declaration + Run). No API key is required.
//
// Usage:
//
//	searchTool := duckduckgotool.New(nil)
//	agent := &agent.Agent{
//	    Tools: []tool.Tool{searchTool},
//	}
package duckduckgotool

import (
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

const (
	defaultMaxResults = 5
	defaultTimeout    = 15 * time.Second
	searchEndpoint    = "https://html.duckduckgo.com/html/"
)

// Config holds configuration for the DuckDuckGo search tool.
type Config struct {
	// MaxResults is the maximum number of search results to return (default: 5).
	MaxResults int
	// HTTPClient allows providing a custom HTTP client (optional).
	HTTPClient *http.Client
	// Timeout for search requests (default: 15s).
	Timeout time.Duration
}

// DuckDuckGoSearch is a web search tool that queries DuckDuckGo.
// It implements the tool.Tool interface and works with any LLM through
// the standard function call mechanism.
type DuckDuckGoSearch struct {
	maxResults int
	httpClient *http.Client
}

// New creates a new DuckDuckGo search tool. Pass nil for default configuration.
func New(cfg *Config) *DuckDuckGoSearch {
	s := &DuckDuckGoSearch{
		maxResults: defaultMaxResults,
	}

	timeout := defaultTimeout

	if cfg != nil {
		if cfg.MaxResults > 0 {
			s.maxResults = cfg.MaxResults
		}
		if cfg.Timeout > 0 {
			timeout = cfg.Timeout
		}
		s.httpClient = cfg.HTTPClient
	}

	if s.httpClient == nil {
		s.httpClient = &http.Client{Timeout: timeout}
	}

	return s
}

// Name implements tool.Tool.
func (s *DuckDuckGoSearch) Name() string {
	return "web_search"
}

// Description implements tool.Tool.
func (s *DuckDuckGoSearch) Description() string {
	return "Searches the web using DuckDuckGo and returns relevant results " +
		"including titles, snippets, and URLs. Use this to find current " +
		"information or research topics."
}

// IsLongRunning implements tool.Tool.
func (s *DuckDuckGoSearch) IsLongRunning() bool {
	return false
}

// Declaration returns the function declaration for the search tool.
func (s *DuckDuckGoSearch) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        s.Name(),
		Description: s.Description(),
		Parameters: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"query": {
					Type:        "STRING",
					Description: "The search query.",
				},
			},
			Required: []string{"query"},
		},
	}
}

// ProcessRequest registers the tool with the LLM request so the model
// can discover and call it.
func (s *DuckDuckGoSearch) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	if req.Tools == nil {
		req.Tools = make(map[string]any)
	}

	name := s.Name()
	if _, ok := req.Tools[name]; ok {
		return fmt.Errorf("duplicate tool: %q", name)
	}
	req.Tools[name] = s

	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}

	decl := s.Declaration()
	if decl == nil {
		return nil
	}

	// Find existing function tool or create a new one.
	var funcTool *genai.Tool
	for _, t := range req.Config.Tools {
		if t != nil && t.FunctionDeclarations != nil {
			funcTool = t
			break
		}
	}

	if funcTool == nil {
		req.Config.Tools = append(req.Config.Tools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{decl},
		})
	} else {
		funcTool.FunctionDeclarations = append(funcTool.FunctionDeclarations, decl)
	}

	return nil
}

// Run executes the web search with the given arguments.
func (s *DuckDuckGoSearch) Run(ctx tool.Context, args any) (map[string]any, error) {
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("expected map[string]any, got %T", args)
	}

	queryRaw, exists := m["query"]
	if !exists {
		return nil, fmt.Errorf("missing required parameter: query")
	}

	query, ok := queryRaw.(string)
	if !ok {
		return nil, fmt.Errorf("query must be a string, got %T", queryRaw)
	}

	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("query must not be empty")
	}

	return s.search(ctx, query)
}

// search performs the actual DuckDuckGo search via the HTML endpoint.
func (s *DuckDuckGoSearch) search(ctx tool.Context, query string) (map[string]any, error) {
	searchURL := searchEndpoint + "?q=" + url.QueryEscape(query)

	req, err := http.NewRequestWithContext(ctx, "GET", searchURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; ADK-Go/1.0)")
	req.Header.Set("Accept", "text/html")
	req.Header.Set("Accept-Language", "en-US,en;q=0.5")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("search request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("search returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	results := parseResults(string(body), s.maxResults)

	return map[string]any{
		"query":   query,
		"results": results,
	}, nil
}

// Regex patterns for parsing DuckDuckGo HTML search results.
var (
	// Matches <a ... class="result__a" ...>Title</a> and captures tag attrs + inner content.
	resultBlockRe = regexp.MustCompile(`(?s)<a\s([^>]*class="result__a"[^>]*)>(.*?)</a>`)
	// Matches <a ... class="result__snippet" ...>Snippet</a> and captures inner content.
	snippetBlockRe = regexp.MustCompile(`(?s)<a\s([^>]*class="result__snippet"[^>]*)>(.*?)</a>`)
	// Extracts href value from tag attributes.
	hrefRe = regexp.MustCompile(`href="([^"]*)"`)
	// Extracts the actual URL from DuckDuckGo's redirect parameter.
	uddgParamRe = regexp.MustCompile(`[?&]uddg=([^&]*)`)
	// Matches HTML tags for stripping.
	htmlTagRe = regexp.MustCompile(`<[^>]*>`)
	// Collapses runs of whitespace.
	whitespaceRe = regexp.MustCompile(`\s+`)
)

// parseResults extracts search results from DuckDuckGo HTML.
func parseResults(htmlContent string, maxResults int) []map[string]any {
	results := make([]map[string]any, 0, maxResults)

	// Find all result link matches with their byte positions.
	indices := resultBlockRe.FindAllStringSubmatchIndex(htmlContent, -1)

	for i, idx := range indices {
		if len(results) >= maxResults {
			break
		}
		// Each match has 3 groups (full, attrs, title) → 6 index values.
		if len(idx) < 6 {
			continue
		}

		attrs := htmlContent[idx[2]:idx[3]] // Group 1: tag attributes
		title := htmlContent[idx[4]:idx[5]] // Group 2: inner content

		// Extract href from the tag attributes.
		hrefMatch := hrefRe.FindStringSubmatch(attrs)
		if len(hrefMatch) < 2 {
			continue
		}

		actualURL := extractURL(hrefMatch[1])
		if actualURL == "" || strings.Contains(actualURL, "duckduckgo.com") {
			continue
		}

		result := map[string]any{
			"title": strings.TrimSpace(stripHTML(title)),
			"url":   actualURL,
		}

		// Look for the snippet between this result link and the next one.
		sectionEnd := len(htmlContent)
		if i+1 < len(indices) {
			sectionEnd = indices[i+1][0]
		}
		section := htmlContent[idx[0]:sectionEnd]

		snippetMatch := snippetBlockRe.FindStringSubmatch(section)
		if len(snippetMatch) >= 3 {
			snippet := strings.TrimSpace(stripHTML(snippetMatch[2]))
			if snippet != "" {
				result["snippet"] = snippet
			}
		}

		results = append(results, result)
	}

	return results
}

// extractURL resolves the actual URL from a DuckDuckGo redirect link.
func extractURL(rawURL string) string {
	// DuckDuckGo wraps result URLs in a redirect: //duckduckgo.com/l/?uddg=ENCODED_URL
	if matches := uddgParamRe.FindStringSubmatch(rawURL); len(matches) >= 2 {
		decoded, err := url.QueryUnescape(matches[1])
		if err == nil {
			return decoded
		}
	}

	// Protocol-relative URL.
	if strings.HasPrefix(rawURL, "//") {
		return "https:" + rawURL
	}

	// Direct URL.
	if strings.HasPrefix(rawURL, "http://") || strings.HasPrefix(rawURL, "https://") {
		return rawURL
	}

	return ""
}

// stripHTML removes HTML tags and decodes HTML entities.
func stripHTML(s string) string {
	s = htmlTagRe.ReplaceAllString(s, "")
	s = html.UnescapeString(s)
	s = whitespaceRe.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}
