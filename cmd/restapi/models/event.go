package models

import "time"

type Event struct {
	ID                 string    `json:"id"`
	Time               time.Time `json:"time"`
	InvocationID       string    `json:"invocation_id"`
	Branch             string    `json:"branch"`
	Author             string    `json:"author"`
	Partial            bool      `json:"partial"`
	LongRunningToolIDs []string  `json:"long_running_tool_ids"`
}
