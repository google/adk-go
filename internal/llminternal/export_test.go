package llminternal

import (
	"google.golang.org/adk/session" 
	"google.golang.org/genai"
)

func BuildContentsCurrentTurnContextOnlyForTest(
	agentName string,
	branch string,
	events []*session.Event,
) ([]*genai.Content, error) {
	return buildContentsCurrentTurnContextOnly(
		agentName,
		branch,
		events,
	)
}