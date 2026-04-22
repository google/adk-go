package llminternal

import "google.golang.org/adk/session"
import "google.golang.org/genai"

// Exportăm funcția privată către pachetul de test extern (llminternal_test)
// Folosim un alias cu literă mare.
func BuildContentsCurrentTurnContextOnlyForTest(agentName, branch string, events []*session.Event) ([]*genai.Content, error) {
	return buildContentsCurrentTurnContextOnly(agentName, branch, events)
}