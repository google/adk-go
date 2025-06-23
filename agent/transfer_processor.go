package agent

import (
	"bytes"
	"context"
	"errors"
	"html/template"
	"log"

	"github.com/google/adk-go"
)

type agentTransferRequestProcesor struct{}

func (p *agentTransferRequestProcesor) Process(ctx context.Context, ic *adk.InvocationContext, req *adk.LLMRequest) error {
	agent := asLLMAgent(ic.Agent)
	if agent == nil {
		return errors.New("not an LLM agent")
	}
	tool, err := agentTransferTool(agent)
	if err != nil {
		return err
	}
	req.AppendTools(tool)
	return nil
}

type agentTransferInput struct {
	TargetAgentName string `json:"target_agent_name"` // Name of the agent to transfer to.
}

type agentTransferOutput struct{}

func agentTransferTool(a *LLMAgent) (*adk.Tool, error) {
	targets := availableTransferTargets(a)
	tmpl, err := template.New("agent_transfer_description").Parse(agentTransferTemplate)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, struct {
		AgentName string
		Parent    adk.Agent
		Targets   []adk.Agent
	}{
		AgentName: a.Name(),
		Parent:    a.ParentAgent,
		Targets:   targets,
	}); err != nil {
		return nil, err
	}

	handler := func(ctx context.Context, input *agentTransferInput) (*agentTransferOutput, error) {
		log.Println("hello world", input)
		return &agentTransferOutput{}, nil
	}
	tool := adk.NewTool("agent_transfer", buf.String(), handler)
	return tool, nil
}

func availableTransferTargets(current *LLMAgent) []adk.Agent {
	var targets []adk.Agent
	targets = append(targets, current.SubAgents...)
	parent := current.ParentAgent
	if parent == nil {
		return targets
	}
	if !current.DisallowTransferToParent {
		targets = append(targets, parent)
	}
	if !current.DisallowTransferToPeers {
		targets = append(targets, parent.Subs()...)
	}
	return targets
}

const agentTransferTemplate = `You have a list of other agents to transfer to:
{{range .Targets}}
Agent name: {{.Name}}
Agent description: {{.Description}}
{{end}}
If you are the best to answer the question according to your description, you
can answer it.
If another agent is better for answering the question according to its
description, call '{{.AgentName}}' function to transfer the
question to that agent. When transfering, do not generate any text other than
the function call.
{{if .Parent}}
Your parent agent is {{.Parent.Name}}. If neither the other agents nor
you are best for answering the question according to the descriptions, transfer
to your parent agent. If you don't have parent agent, try answer by yourself.
{{end}}
`
