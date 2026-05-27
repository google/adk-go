package workflowinternal

type TaskRequest struct {
	AgentName string
	Input     map[string]any
}

type TaskResult struct {
	Output any
}

type DefaultTaskInput struct {
	Goal       string
	Background string
}
