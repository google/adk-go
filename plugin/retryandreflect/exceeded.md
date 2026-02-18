
The tool `{{.ToolName}}` has failed consecutively {self.max_retries} times and the retry limit has been exceeded.

**Last Error:**
```
{{.ErrorDetails}}
```

**Last Arguments Used:**
```json
{{.ArgsSummary}}
```

**Final Instruction:**
**Do not attempt to use the `{{.ToolName}}` tool again for this task.** You must now try a different approach. Acknowledge the failure and devise a new strategy, potentially using other available tools or informing the user that the task cannot be completed.
