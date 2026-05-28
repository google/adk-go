package workflowinternal

import (
	"google.golang.org/adk/agent"
	"google.golang.org/adk/tool"
)

func NewSingleTurnTool(a agent.Agent) (tool.Tool, error) {
	// TODO: implement
	return nil, nil
}

//func (t *singleTurnTool) Run(ctx tool.Context, args any) (map[string]any, error) {
// nc, ok := workflow.NodeContextFromGoContext(ctx)
// node, err := workflow.NewAgentNode(t.agent, workflow.NodeConfig{})
// out, err := workflow.RunNode[any](nc, node, nodeInput)
//}
