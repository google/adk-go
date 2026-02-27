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

package llmjudge

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"google.golang.org/adk/evaluation"
)

// ResponseParser extracts structured data from LLM judge responses.
type ResponseParser struct {
	scorePattern   *regexp.Regexp
	verdictPattern *regexp.Regexp
	rubricPattern  *regexp.Regexp
}

// NewResponseParser creates a new response parser.
func NewResponseParser() *ResponseParser {
	return &ResponseParser{
		// Matches patterns like "Score: 4.5", "score = 3", "Rating: 5/5"
		scorePattern: regexp.MustCompile(`(?i)(?:score|rating)[:=\s]+(\d+\.?\d*)`),

		// Matches yes/no verdicts
		verdictPattern: regexp.MustCompile(`(?i)\b(yes|no)\b`),

		// Matches rubric responses like "RubricID: yes" or "criterion_1: no"
		rubricPattern: regexp.MustCompile(`(?i)([a-z0-9_-]+)[:=\s]+(yes|no)`),
	}
}

// ParseScore extracts a numeric score from the LLM response.
// Expected format: "Score: X" or "Rating: X/Y" or free-form text containing score.
func (p *ResponseParser) ParseScore(response string) (float64, error) {
	matches := p.scorePattern.FindStringSubmatch(response)
	if len(matches) < 2 {
		return 0, fmt.Errorf("no score found in response: %s", response)
	}

	score, err := strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return 0, fmt.Errorf("failed to parse score %q: %w", matches[1], err)
	}

	return score, nil
}

// ParseVerdict extracts a yes/no verdict from the LLM response.
func (p *ResponseParser) ParseVerdict(response string) (string, error) {
	matches := p.verdictPattern.FindStringSubmatch(response)
	if len(matches) < 2 {
		return "", fmt.Errorf("no verdict found in response: %s", response)
	}

	verdict := strings.ToLower(matches[1])
	return verdict, nil
}

// ParseRubricScores extracts rubric verdicts from the LLM response.
// Expected format: Multiple lines like "rubric_id: yes/no"
func (p *ResponseParser) ParseRubricScores(response string, rubrics map[string]evaluation.Rubric) (map[string]evaluation.RubricScore, error) {
	if len(rubrics) == 0 {
		return nil, fmt.Errorf("no rubrics provided")
	}

	rubricScores := make(map[string]evaluation.RubricScore)

	// Find all rubric verdicts in the response
	matches := p.rubricPattern.FindAllStringSubmatch(response, -1)
	if len(matches) == 0 {
		return nil, fmt.Errorf("no rubric verdicts found in response: %s", response)
	}

	for _, match := range matches {
		if len(match) < 3 {
			continue
		}

		rubricID := match[1]
		verdict := strings.ToLower(match[2])

		// Check if this rubric ID exists in our rubrics
		if _, exists := rubrics[rubricID]; exists {
			score := 0.0
			if verdict == "yes" {
				score = 1.0
			}

			rubricScores[rubricID] = evaluation.RubricScore{
				RubricID: rubricID,
				Score:    score,
				Verdict:  verdict,
			}
		} else {
			// Try fuzzy matching (case-insensitive)
			for rID := range rubrics {
				if strings.EqualFold(rID, rubricID) {
					score := 0.0
					if verdict == "yes" {
						score = 1.0
					}

					rubricScores[rID] = evaluation.RubricScore{
						RubricID: rID,
						Score:    score,
						Verdict:  verdict,
					}
					break
				}
			}
		}
	}

	if len(rubricScores) == 0 {
		return nil, fmt.Errorf("no matching rubrics found in response")
	}

	return rubricScores, nil
}

// ParseExplanation extracts an explanation from the LLM response.
// This looks for common explanation patterns.
func (p *ResponseParser) ParseExplanation(response string) string {
	// Common explanation markers
	markers := []string{
		"Explanation:",
		"Reasoning:",
		"Justification:",
		"Because:",
	}

	for _, marker := range markers {
		if idx := strings.Index(response, marker); idx != -1 {
			explanation := strings.TrimSpace(response[idx+len(marker):])
			// Take first sentence or up to 200 chars
			if dotIdx := strings.Index(explanation, "."); dotIdx != -1 {
				explanation = explanation[:dotIdx+1]
			}
			if len(explanation) > 200 {
				explanation = explanation[:200] + "..."
			}
			return explanation
		}
	}

	// If no marker found, return first 100 chars
	if len(response) > 100 {
		return response[:100] + "..."
	}
	return response
}

// ParseClassification extracts a classification label from the LLM response.
// Used for hallucination detection (Supported, Unsupported, Contradictory, etc.)
func (p *ResponseParser) ParseClassification(response string, validLabels []string) (string, error) {
	responseLower := strings.ToLower(response)

	for _, label := range validLabels {
		if strings.Contains(responseLower, strings.ToLower(label)) {
			return label, nil
		}
	}

	return "", fmt.Errorf("no valid classification found in response: %s", response)
}
