package runner

import (
	"context"

	"github.com/google/adk-go"

	"google.golang.org/genai"
)

type Runner struct {
	AppName        string
	Agent          adk.Agent
	SessionService adk.SessionService
}

// Run runs the agent.
func (r *Runner) Run(ctx context.Context, userID, sessionID string, msg *genai.Content, cfg *adk.AgentRunConfig) (adk.EventStream, error) {
	// TODO(hakim): we need to validate whether cfg is compatible with the Agent.
	//   see adk-python/src/google/adk/runners.py Runner._new_invocation_context.
	//
	// For example, support_cfc requires Agent to be LLMAgent.
	// Note that checking that directly in this package results in circular dependency.
	// Options to consider:
	//     - Move Runner to a separate package (runner imports adk, agent. agent imports adk).
	//     - Require Agent.Validate method.
	//     - Wait until Agent.Run is called.
	/*
		// TODO: setup tracer.
		session, err := r.SessionService.Create(r.AppName, userID, nil)
		if err != nil {
			return nil, err
		}
		invocationCtx := r.newInvocationContext(ctx, session, msg, cfg)
		...
	*/
	panic("unimplemened")
}
